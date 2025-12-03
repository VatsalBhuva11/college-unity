using System;
using System.Net.Sockets;
using System.Collections;
using UnityEngine;

// FlockingDDPG.cs (fixed variable naming to avoid C# shadowing errors)
// Attach to a Unity GameObject. Set 'drones' (array length NUM_DRONES) and 'target' in Inspector.
// Protocol is unchanged.

public class FlockingDDPG : MonoBehaviour
{
    public GameObject[] drones; // assign NUM_DRONES in inspector
    public Transform target;
    public int NUM_DRONES = 3;
    private TcpClient client;
    private NetworkStream stream;

    private const int STATE_DIM = 19; // same as Python STATE_DIM
    private const int ACTION_DIM = 2;
    private const int FULL_DIM = STATE_DIM + 2; // state + reward + flag

    private Vector3[] initialPositions;
    private Quaternion[] initialRotations;
    private float[] prevDistToTarget;
    private bool[] droneDone;

    // TUNING: Use a yaw-rate (degrees per second) rather than a tiny per-step deg.
    public float MAX_SPEED = 10f;                 // m/s for throttle
    public float MAX_TURN_RATE_DEG_PER_SEC = 90f; // degrees per second for steering
    public float RAYCAST_RANGE = 40f;

    // Optional debug: enable for verbose logs
    public bool debugLogs = false;

    void Start()
    {
        if (drones == null || drones.Length != NUM_DRONES)
        {
            Debug.LogError("Please assign exactly NUM_DRONES drones in inspector.");
            return;
        }

        prevDistToTarget = new float[NUM_DRONES];
        droneDone = new bool[NUM_DRONES];
        initialPositions = new Vector3[NUM_DRONES];
        initialRotations = new Quaternion[NUM_DRONES];
        StoreInitialPositions();
        ConnectToPython();

        // send initial state once (rewards=0 flags=0)
        float[] zerosR = new float[NUM_DRONES];
        int[] zerosF = new int[NUM_DRONES];
        SendStatesToPython(zerosR, zerosF);

        StartCoroutine(CommunicationLoop());
    }

    void StoreInitialPositions()
    {
        for (int i = 0; i < NUM_DRONES; i++)
        {
            initialPositions[i] = drones[i].transform.position;
            initialRotations[i] = drones[i].transform.rotation;
            prevDistToTarget[i] = Vector3.Distance(drones[i].transform.position, target.position);
            droneDone[i] = false;
        }
    }

    void ConnectToPython()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 5555);
            stream = client.GetStream();
            Debug.Log("Connected to Python trainer.");
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to connect to Python: " + e.Message);
        }
    }

    float[] GetDroneState(GameObject drone)
    {
        Rigidbody rb = drone.GetComponent<Rigidbody>();
        float[] state = new float[STATE_DIM];

        // yaw in degrees
        state[0] = drone.transform.eulerAngles.y;

        // velocity magnitude
        state[1] = rb != null ? rb.velocity.magnitude : 0f;

        Vector3 toTarget = target.position - drone.transform.position;
        state[2] = Vector3.SignedAngle(drone.transform.forward, toTarget, Vector3.up);
        state[3] = toTarget.magnitude;

        // find two nearest neighbors (excluding self)
        float min1 = float.MaxValue, min2 = float.MaxValue;
        GameObject nearestObj1 = null, nearestObj2 = null;

        foreach (GameObject other in drones)
        {
            if (other == drone) continue;
            Vector3 toOther = other.transform.position - drone.transform.position;
            float dist = toOther.magnitude;
            if (dist < min1)
            {
                min2 = min1;
                nearestObj2 = nearestObj1;
                min1 = dist;
                nearestObj1 = other;
            }
            else if (dist < min2)
            {
                min2 = dist;
                nearestObj2 = other;
            }
        }

        if (nearestObj1 != null)
        {
            Vector3 toN1 = nearestObj1.transform.position - drone.transform.position;
            state[4] = Vector3.SignedAngle(drone.transform.forward, toN1, Vector3.up);
            state[5] = min1;
        }
        else
        {
            state[4] = 0f; state[5] = 100f;
        }

        if (nearestObj2 != null)
        {
            Vector3 toN2 = nearestObj2.transform.position - drone.transform.position;
            state[6] = Vector3.SignedAngle(drone.transform.forward, toN2, Vector3.up);
            state[7] = min2;
        }
        else
        {
            state[6] = 0f; state[7] = 100f;
        }

        // 9 raycasts around
        for (int i = 0; i < 9; i++)
        {
            Vector3 dir = Quaternion.Euler(0, i * 40f - 180f, 0) * drone.transform.forward;
            RaycastHit hit;
            if (Physics.Raycast(drone.transform.position, dir, out hit, RAYCAST_RANGE))
            {
                state[8 + i] = hit.distance;
            }
            else
            {
                state[8 + i] = RAYCAST_RANGE;
            }
        }

        // alignment with nearest neighbours: dot product (-1..1)
        if (nearestObj1 != null)
        {
            Vector3 f1 = drone.transform.forward.normalized;
            Vector3 fN1 = nearestObj1.transform.forward.normalized;
            state[17] = Vector3.Dot(f1, fN1);
        }
        else state[17] = 0f;

        if (nearestObj2 != null)
        {
            Vector3 f1 = drone.transform.forward.normalized;
            Vector3 fN2 = nearestObj2.transform.forward.normalized;
            state[18] = Vector3.Dot(f1, fN2);
        }
        else state[18] = 0f;

        return state;
    }

    float CalculateReward(int idx, float[] state, int flag, float[] actions)
    {
        float curDist = state[3];
        float prev = prevDistToTarget[idx];
        float reward = (prev - curDist) * 30f;

        if (flag == 1) reward += 200f;
        else if (flag == 2) reward -= 50f;

        float minDist1 = state[5];
        if (minDist1 < 5f) reward -= 5f;
        else if (minDist1 >= 10f && minDist1 <= 30f) reward += 1f;

        float minObstacle = RAYCAST_RANGE;
        for (int i = 8; i <= 16; i++)
        {
            if (state[i] < minObstacle) minObstacle = state[i];
        }
        if (minObstacle < 10f) reward -= 2f * (10f - minObstacle);

        reward -= 0.2f;
        if (actions != null)
        {
            float steer = actions[0];
            reward -= Mathf.Abs(steer) * 0.1f;
        }
        float vel = state[1];
        if (vel < 0.5f) reward -= 1.0f;

        return reward;
    }

    void SendStatesToPython(float[] rewards, int[] flags)
    {
        int packetSize = FULL_DIM; // 21
        float[] outArr = new float[NUM_DRONES * packetSize];
        for (int i = 0; i < NUM_DRONES; i++)
        {
            float[] st = GetDroneState(drones[i]);
            Array.Copy(st, 0, outArr, i * packetSize, STATE_DIM);
            outArr[i * packetSize + STATE_DIM] = rewards[i];
            outArr[i * packetSize + STATE_DIM + 1] = flags[i];
        }
        byte[] bytes = new byte[outArr.Length * 4];
        Buffer.BlockCopy(outArr, 0, bytes, 0, bytes.Length);
        stream.Write(bytes, 0, bytes.Length);
        stream.Flush();
    }

    float[] ReceiveActionsFromPython()
    {
        int expected = NUM_DRONES * ACTION_DIM * 4;
        byte[] data = new byte[expected];
        int read = 0;
        while (read < expected)
        {
            int r = stream.Read(data, read, expected - read);
            if (r == 0)
            {
                Debug.LogError("Python closed connection");
                return null;
            }
            read += r;
        }
        float[] actions = new float[NUM_DRONES * ACTION_DIM];
        Buffer.BlockCopy(data, 0, actions, 0, expected);
        return actions;
    }

    void ApplyActionsToDrones(float[] actions)
    {
        if (actions == null) return;

        for (int i = 0; i < NUM_DRONES; i++)
        {
            if (droneDone[i]) continue;

            prevDistToTarget[i] = Vector3.Distance(drones[i].transform.position, target.position);

            float steering_norm = actions[i * ACTION_DIM + 0];
            float throttle_norm = actions[i * ACTION_DIM + 1];

            steering_norm = Mathf.Clamp(steering_norm, -1f, 1f);
            throttle_norm = Mathf.Clamp(throttle_norm, -1f, 1f);

            ApplyActionToDrone(drones[i], steering_norm, throttle_norm);
        }
    }

    // FixedUpdate-friendly physics rotation; variable names uniquely named to avoid shadowing
    void ApplyActionToDrone(GameObject drone, float steering_norm, float throttle_norm)
    {
        Rigidbody rb = drone.GetComponent<Rigidbody>();
        if (rb == null)
        {
            // fallback: apply transform changes if no Rigidbody
            float yawDeltaNoRb = steering_norm * MAX_TURN_RATE_DEG_PER_SEC * Time.fixedDeltaTime;
            drone.transform.Rotate(0f, yawDeltaNoRb, 0f, Space.World);
            Vector3 desiredVelNoRb = drone.transform.forward * (throttle_norm * MAX_SPEED);
            drone.transform.position += desiredVelNoRb * Time.fixedDeltaTime;
            return;
        }

        // Check common issues (informational)
        if (rb.constraints != RigidbodyConstraints.None)
        {
            if ((rb.constraints & RigidbodyConstraints.FreezeRotationY) != 0)
            {
                Debug.LogWarning($"Rigidbody on {drone.name} has FreezeRotationY — rotation will be prevented.");
            }
        }

        // rotation using yaw rate -> delta this physics step
        float yawRateDegPerSec = steering_norm * MAX_TURN_RATE_DEG_PER_SEC;
        float yawDeltaRb = yawRateDegPerSec * Time.fixedDeltaTime; // degrees for this physics step

        Quaternion deltaRot = Quaternion.Euler(0f, yawDeltaRb, 0f);
        Quaternion newRot = rb.rotation * deltaRot;
        rb.MoveRotation(newRot);

        // throttle: target forward speed (based on Rigidbody rotation)
        Vector3 desiredVelRb = (rb.rotation * Vector3.forward) * (throttle_norm * MAX_SPEED);
        Vector3 accel = desiredVelRb - rb.velocity;

        // clamp acceleration for stability
        float maxAccel = 20f;
        if (accel.magnitude > maxAccel) accel = accel.normalized * maxAccel;

        rb.AddForce(accel, ForceMode.Acceleration);
    }

    void ResetDrones()
    {
        for (int i = 0; i < NUM_DRONES; i++)
        {
            Rigidbody rb = drones[i].GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.velocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }
            drones[i].transform.position = initialPositions[i];
            drones[i].transform.rotation = initialRotations[i];

            // small random yaw for exploration
            float randomY = UnityEngine.Random.Range(-30f, 30f);
            drones[i].transform.rotation = Quaternion.Euler(0, drones[i].transform.eulerAngles.y + randomY, 0);

            droneDone[i] = false;

            prevDistToTarget[i] = Vector3.Distance(drones[i].transform.position, target.position);

            var cd = drones[i].GetComponent<CollisionDetector>();
            if (cd != null) cd.HasCollided = false;
        }
    }

    bool DroneAtTarget(int index)
    {
        return Vector3.Distance(drones[index].transform.position, target.position) <= 2.0f;
    }

    void CheckTerminationFlags()
    {
        bool allDone = true;
        for (int i = 0; i < NUM_DRONES; i++)
        {
            var cd = drones[i].GetComponent<CollisionDetector>();
            if (cd != null && cd.HasCollided) droneDone[i] = true;
            if (!droneDone[i] && DroneAtTarget(i)) droneDone[i] = true;
            if (!droneDone[i]) allDone = false;
        }
    }

    System.Collections.IEnumerator CommunicationLoop()
    {
        while (true)
        {
            float[] actions = ReceiveActionsFromPython();
            if (actions == null) yield break;

            // Reset signal
            if (actions.Length > 0 && Mathf.Approximately(actions[0], -99f))
            {
                ResetDrones();
                // send initial zero state to python
                SendStatesToPython(new float[NUM_DRONES], new int[NUM_DRONES]);
                yield return null;
                continue;
            }

            ApplyActionsToDrones(actions);

            // Wait for Unity physics step
            yield return new WaitForFixedUpdate();

            // Evaluate termination after physics update
            CheckTerminationFlags();

            // Prepare rewards and flags for next state
            float[] rewards = new float[NUM_DRONES];
            int[] flags = new int[NUM_DRONES];
            for (int i = 0; i < NUM_DRONES; i++)
            {
                float[] st = GetDroneState(drones[i]);
                int flag = 0;
                var cd = drones[i].GetComponent<CollisionDetector>();
                bool collided = (cd != null && cd.HasCollided);
                if (DroneAtTarget(i))
                {
                    flag = 1;
                    droneDone[i] = true;
                }
                else if (droneDone[i] || collided)
                {
                    flag = 2;
                    droneDone[i] = true;
                }
                flags[i] = flag;

                float[] droneAction = new float[ACTION_DIM];
                Array.Copy(actions, i * ACTION_DIM, droneAction, 0, ACTION_DIM);

                rewards[i] = CalculateReward(i, st, flag, droneAction);

                if (debugLogs)
                {
                    Debug.Log($"Drone {i} reward {rewards[i]:F2}, flag {flag}, vel {st[1]:F2}, distT {st[3]:F2}");
                }
            }

            // Send next state + rewards + flags
            SendStatesToPython(rewards, flags);

            // If termination occurred for the whole episode (all Done), reset after sending
            bool allDone = true;
            for (int i = 0; i < NUM_DRONES; i++) if (!droneDone[i]) allDone = false;
            if (allDone)
            {
                // Reset after sending final states
                ResetDrones();
                SendStatesToPython(new float[NUM_DRONES], new int[NUM_DRONES]);
            }
        }
    }

    void OnApplicationQuit()
    {
        if (stream != null) stream.Close();
        if (client != null) client.Close();
    }
}

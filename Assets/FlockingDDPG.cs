using System;
using System.Net.Sockets;
using System.Collections;
using UnityEngine;

public class FlockingDrones : MonoBehaviour
{
    public GameObject[] drones; // Assign 3 drones in Unity Inspector
    public Transform target;
    private TcpClient client;
    private NetworkStream stream;
    private const int STATE_DIM = 17; // State dimension per drone
    private const int ACTION_DIM = 2; // Action dimension per drone
    private const int NUM_DRONES = 3; // Number of drones

    private Vector3[] initialDronePositions;
    private Quaternion[] initialDroneRotations;

    private bool episodeTerminated = false;

    void Start()
    {
        ConnectToPython();
        StoreInitialPositions();
        StartCoroutine(CommunicationLoop());
    }

    void ResetDrones()
    {
        for (int i = 0; i < NUM_DRONES; i++)
        {
            Rigidbody rb = drones[i].GetComponent<Rigidbody>();
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            drones[i].transform.position = initialDronePositions[i];
            drones[i].transform.rotation = initialDroneRotations[i];
            CollisionDetector collisionDetector = drones[i].GetComponent<CollisionDetector>();
            if (collisionDetector != null)
            {
                collisionDetector.HasCollided = false;
            }
        }
    }

    void StoreInitialPositions()
    {
        initialDronePositions = new Vector3[NUM_DRONES];
        initialDroneRotations = new Quaternion[NUM_DRONES];
        for (int i = 0; i < NUM_DRONES; i++)
        {
            initialDronePositions[i] = drones[i].transform.position;
            initialDroneRotations[i] = drones[i].transform.rotation;
        }
    }

    void ConnectToPython()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 5555);
            stream = client.GetStream();
            Debug.Log("Connected to Python.");
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to connect: " + e.Message);
        }
    }

    // Returns the current state (17 floats) of the given drone.
    float[] GetDroneState(GameObject drone)
    {
        Rigidbody rb = drone.GetComponent<Rigidbody>();
        float[] state = new float[STATE_DIM];
        state[0] = drone.transform.eulerAngles.y;
        state[1] = rb.velocity.magnitude;
        Vector3 toTarget = target.position - drone.transform.position;
        state[2] = Vector3.Angle(drone.transform.forward, toTarget);
        state[3] = toTarget.magnitude;
        float minDist1 = float.MaxValue, minDist2 = float.MaxValue;
        float angle1 = 0, angle2 = 0;
        foreach (GameObject other in drones)
        {
            if (other == drone) continue;
            Vector3 toOther = other.transform.position - drone.transform.position;
            float dist = toOther.magnitude;
            float angle = Vector3.Angle(drone.transform.forward, toOther);
            if (dist < minDist1)
            {
                minDist2 = minDist1;
                angle2 = angle1;
                minDist1 = dist;
                angle1 = angle;
            }
            else if (dist < minDist2)
            {
                minDist2 = dist;
                angle2 = angle;
            }
        }
        state[4] = angle1;
        state[5] = minDist1;
        state[6] = angle2;
        state[7] = minDist2;
        for (int i = 0; i < 9; i++)
        {
            Vector3 direction = Quaternion.Euler(0, i * 40 - 180, 0) * drone.transform.forward;
            RaycastHit hit;
            if (Physics.Raycast(drone.transform.position, direction, out hit, 40f))
            {
                state[8 + i] = hit.distance;
            }
            else
            {
                state[8 + i] = 40f;
            }
        }
        return state;
    }

    // Sends the states of all drones to Python.
    void SendStatesToPython()
    {
        float[] allStates = new float[NUM_DRONES * (STATE_DIM + 1)]; // +1 for termination flag
        for (int i = 0; i < NUM_DRONES; i++)
        {
            float[] state = GetDroneState(drones[i]);
            Array.Copy(state, 0, allStates, i * (STATE_DIM + 1), STATE_DIM);
            allStates[i * (STATE_DIM + 1) + STATE_DIM] = episodeTerminated ? 1 : 0; // Termination flag
        }
        byte[] data = new byte[allStates.Length * 4];
        foreach (float state in allStates)
{
    Debug.Log(state);
}
        Buffer.BlockCopy(allStates, 0, data, 0, data.Length);
        stream.Write(data, 0, data.Length);
        stream.Flush();
    }

    // Receives actions for all drones from Python.
    float[] ReceiveActionsFromPython()
    {
        byte[] data = new byte[NUM_DRONES * ACTION_DIM * 4];
        int bytesRead = 0;
        while (bytesRead < data.Length)
        {
            int read = stream.Read(data, bytesRead, data.Length - bytesRead);
            if (read == 0) break;
            bytesRead += read;
        }
        float[] actions = new float[NUM_DRONES * ACTION_DIM];
        Buffer.BlockCopy(data, 0, actions, 0, data.Length);
        return actions;
    }

    // Applies actions to all drones.
    void ApplyActionsToDrones(float[] actions)
    {
        for (int i = 0; i < NUM_DRONES; i++)
        {
            float[] action = new float[ACTION_DIM];
            Array.Copy(actions, i * ACTION_DIM, action, 0, ACTION_DIM);
            ApplyActionToDrone(drones[i], action);
        }
    }

    void ApplyActionToDrone(GameObject drone, float[] action)
    {
        Rigidbody rb = drone.GetComponent<Rigidbody>();
        float steering = action[0];
        float throttle = action[1];
        float maxSteeringAngle = Mathf.PI / 4;
        float targetTurnAngle = steering * maxSteeringAngle * Mathf.Rad2Deg;
        StartCoroutine(RotateDroneOverTime(drone, targetTurnAngle, 0.05f));
        Vector3 force = drone.transform.forward * throttle * 10f;
        rb.AddForce(force, ForceMode.Acceleration);
    }

    private IEnumerator RotateDroneOverTime(GameObject drone, float targetAngle, float duration)
    {
        float elapsedTime = 0f;
        Quaternion startRotation = drone.transform.rotation;
        Quaternion targetRotation = startRotation * Quaternion.Euler(0, targetAngle, 0);
        while (elapsedTime < duration)
        {
            drone.transform.rotation = Quaternion.Slerp(startRotation, targetRotation, elapsedTime / duration);
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        drone.transform.rotation = targetRotation;
    }

    // Checks termination conditions (collision or target reached).
    bool CheckTerminationConditions()
    {
        if (CheckCollision() || CheckTargetReached())
        {
            Debug.Log("Episode Ended: termination condition met.");
            episodeTerminated = true;
            return true;
        }
        return false;
    }

    bool CheckCollision()
    {
        foreach (GameObject drone in drones)
        {
            if (drone.GetComponent<CollisionDetector>().HasCollided)
                return true;
        }
        return false;
    }

    bool CheckTargetReached()
    {
        foreach (GameObject drone in drones)
        {
            if (Vector3.Distance(drone.transform.position, target.position) > 2.0f)
                return false;
        }
        return true;
    }


    IEnumerator CommunicationLoop()
    {
        while (true)
        {
            // Send states to Python
            SendStatesToPython();

            // Receive actions from Python
            float[] actions = ReceiveActionsFromPython();

            // Apply actions to drones
            ApplyActionsToDrones(actions);
            
            // Send updated states to Python
            SendStatesToPython();

            // Check termination conditions
            if (CheckTerminationConditions())
            {
                ResetDrones();
                episodeTerminated = false;
            }

            yield return new WaitForFixedUpdate();
        }
    }

    void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
    }
}

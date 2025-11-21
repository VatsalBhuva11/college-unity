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
    private float[] prevDistToTarget; // To calculate reward

    private int terminationFlag = 0; // 0: running, 1: target reached, 2: collision
    public float MAX_SPEED = 10f;          // tune based on environment
    public float MAX_STEERING_DEG = 10f;   // max steering angle in degrees


    void Start()
    {
        prevDistToTarget = new float[NUM_DRONES];
        ConnectToPython();
        StoreInitialPositions();
        
        // Send initial state ONCE with reward=0, done=0
        float[] initialRewards = new float[NUM_DRONES]; // All zeros
        SendStatesToPython(initialRewards);
        
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
        terminationFlag = 0; // Reset termination flag
        
        // Reset prev distances
        for (int i = 0; i < NUM_DRONES; i++) {
            prevDistToTarget[i] = Vector3.Distance(drones[i].transform.position, target.position);
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
            prevDistToTarget[i] = Vector3.Distance(drones[i].transform.position, target.position);
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
    
    float CalculateReward(int droneIndex, float[] currentState, int doneFlag, float[] actions)
    {
        float curDist = currentState[3]; // Distance to target is at index 3
        float prevDist = prevDistToTarget[droneIndex];
        
        // 1. Base movement reward
        float reward = (prevDist - curDist) * 100f;
        
        // 2. Target reached / Collision
        if (doneFlag == 1) reward += 100f;
        else if (doneFlag == 2) reward -= 100f;
        
        // 3. Flocking
        float minDist1 = currentState[5]; // Min dist neighbor is at index 5
        if (minDist1 < 5f) reward -= 20f;
        else if (minDist1 >= 10f && minDist1 <= 30f) reward += 5f;
        
        // 4. Obstacles indices 8 to 16 (9 rays)
        float minObstacle = 40f;
        for (int i = 8; i <= 16; i++) {
            if (currentState[i] < minObstacle) minObstacle = currentState[i];
        }
        if (minObstacle < 10f) reward -= 10f * (10f - minObstacle);
        
        // 5. Step penalty
        reward -= 0.5f; 
        
        // 6. Action Smoothness / Steering Penalty
        // actions[0] is steering ([-1,1])
        // Penalize sharp turns to avoid "aggressive rotating"
        if (actions != null)
        {
            float steering = actions[0];
            reward -= Mathf.Abs(steering) * 2.0f; 
        }
        
        // 7. Movement incentive
        // If velocity (index 1) is very low, penalize slightly
        float velocity = currentState[1];
        if (velocity < 0.5f) reward -= 1.0f;

        return reward;
    }

    // Sends the states of all drones to Python with individual termination flags and rewards
    // Protocol: [State(17), Reward(1), Flag(1)] * NUM_DRONES
    void SendStatesToPython(float[] rewards)
    {
        int packetSize = STATE_DIM + 2; 
        float[] allData = new float[NUM_DRONES * packetSize];
        
        for (int i = 0; i < NUM_DRONES; i++)
        {
            float[] state = GetDroneState(drones[i]);
            Array.Copy(state, 0, allData, i * packetSize, STATE_DIM);
            
            allData[i * packetSize + STATE_DIM] = rewards[i];
            
            // Individual termination flag per drone
            int droneTerminationFlag = 0;
            if (terminationFlag == 1) // Target reached (all drones)
            {
                droneTerminationFlag = 1;
            }
            else if (terminationFlag == 2) // Collision occurred
            {
                // Mark as collision
                droneTerminationFlag = 2;
            }
            
            allData[i * packetSize + STATE_DIM + 1] = (float)droneTerminationFlag;
        }
        
        byte[] data = new byte[allData.Length * 4];
        Buffer.BlockCopy(allData, 0, data, 0, data.Length);
        stream.Write(data, 0, data.Length);
        stream.Flush();
    }

    // Receives actions for all drones from Python.
    float[] ReceiveActionsFromPython()
    {
        int expected = NUM_DRONES * ACTION_DIM * 4;
        byte[] data = new byte[expected];
        int bytesRead = 0;
        while (bytesRead < expected)
        {
            int read = stream.Read(data, bytesRead, expected - bytesRead);
            if (read == 0)
            {
                Debug.LogError("Python closed connection");
                return null;
            }
            bytesRead += read;
        }
        float[] actions = new float[NUM_DRONES * ACTION_DIM];
        Buffer.BlockCopy(data, 0, actions, 0, expected);
        return actions;
    }


    // Applies actions to all drones.
    void ApplyActionsToDrones(float[] actions)
    {
        if (actions == null) return;  // handle disconnect

        for (int i = 0; i < NUM_DRONES; i++)
        {
            // store prev dist for reward calc
            prevDistToTarget[i] = Vector3.Distance(drones[i].transform.position, target.position);
            
            float steering_norm = actions[i * ACTION_DIM + 0]; // [-1,1]
            float throttle_norm = actions[i * ACTION_DIM + 1]; // [0,1] by our protocol

            // clip to safe ranges just in case
            steering_norm = Mathf.Clamp(steering_norm, -1f, 1f);
            throttle_norm = Mathf.Clamp(throttle_norm, 0f, 1f); // ensures non-negative

            ApplyActionToDrone(drones[i], steering_norm, throttle_norm);
        }
    }

    void ApplyActionToDrone(GameObject drone, float steering_norm, float throttle_norm)
    {
        Rigidbody rb = drone.GetComponent<Rigidbody>();
        // Steering: convert [-1,1] -> [-MAX_STEERING_DEG, MAX_STEERING_DEG]
        float targetTurnAngle = steering_norm * MAX_STEERING_DEG;

        // throttle_norm is [0,1] -> scale to speed/accel
        float appliedForwardSpeed = throttle_norm * MAX_SPEED;

        // rotate over time (as before)
        StartCoroutine(RotateDroneOverTime(drone, targetTurnAngle, 0.05f));

        // apply forward acceleration toward desired forward speed
        // Option 1: add acceleration proportional to desired speed:
        Vector3 desiredVelocity = drone.transform.forward * appliedForwardSpeed;
        Vector3 accel = (desiredVelocity - rb.velocity);
        // Optionally cap acceleration
        float maxAccel = 10f;
        if (accel.magnitude > maxAccel) accel = accel.normalized * maxAccel;
        rb.AddForce(accel, ForceMode.Acceleration);
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
    void CheckTerminationConditions()
    {
        if (CheckCollision())
        {
            Debug.Log("Episode Ended: Collision detected.");
            terminationFlag = 2; // Collision
        }
        else if (CheckTargetReached())
        {
            Debug.Log("Episode Ended: Target reached.");
            terminationFlag = 1; // Target reached
        }
        else
        {
            terminationFlag = 0; // Still running
        }
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
            // 1. Receive actions from Python
            float[] actions = ReceiveActionsFromPython();

            // 2. Apply actions to drones (and store prev dists)
            ApplyActionsToDrones(actions);

            // 3. Wait for physics update
            yield return new WaitForFixedUpdate();

            // 4. Check termination conditions AFTER physics update
            CheckTerminationConditions();
            
            // 5. Calculate Rewards based on new state
            float[] rewards = new float[NUM_DRONES];
            for(int i=0; i<NUM_DRONES; i++) {
                float[] state = GetDroneState(drones[i]);
                
                // Extract action for this drone to pass to reward function
                float[] droneAction = new float[ACTION_DIM];
                if (actions != null)
                    Array.Copy(actions, i * ACTION_DIM, droneAction, 0, ACTION_DIM);
                else 
                    droneAction = null;

                // Flag is global (terminationFlag)
                rewards[i] = CalculateReward(i, state, terminationFlag, droneAction);
            }
            
            // 6. Send the NEXT state, Reward, and Flag
            SendStatesToPython(rewards);

            // 7. If episode ended, reset and send initial state for next episode
            if (terminationFlag != 0)
            {
                ResetDrones();
                // Send new initial state (Rewards=0)
                SendStatesToPython(new float[NUM_DRONES]);
            }
        }
    }

    void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
    }
}

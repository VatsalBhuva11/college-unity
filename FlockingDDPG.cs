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
    private const int STATE_DIM = 19; // State dimension per drone (Increased for alignment)
    private const int ACTION_DIM = 2; // Action dimension per drone
    private const int NUM_DRONES = 3; // Number of drones

    private Vector3[] initialDronePositions;
    private Quaternion[] initialDroneRotations;
    private float[] prevDistToTarget; // To calculate reward
    private bool[] droneDone;

    private int terminationFlag = 0; // 0: running, 1: target reached, 2: collision
    public float MAX_SPEED = 10f;          // tune based on environment
    public float MAX_STEERING_DEG = 1.5f;    // max steering angle in degrees (Reduced to prevent spinning)


    void Start()
    {
        prevDistToTarget = new float[NUM_DRONES];
        droneDone = new bool[NUM_DRONES];
        droneDone = new bool[NUM_DRONES];
        ConnectToPython();
        StoreInitialPositions();
        
        // Send initial state ONCE with reward=0, done=0
        float[] initialRewards = new float[NUM_DRONES]; // All zeros
        int[] initialFlags = new int[NUM_DRONES];
        SendStatesToPython(initialRewards, initialFlags);
        
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
            
            // Randomize rotation slightly to encourage diverse exploration
            float randomY = UnityEngine.Random.Range(-180f, 180f);
            drones[i].transform.rotation = Quaternion.Euler(0, randomY, 0);
            droneDone[i] = false;
            
            CollisionDetector collisionDetector = drones[i].GetComponent<CollisionDetector>();
            if (collisionDetector != null)
            {
                collisionDetector.HasCollided = false;
            }

            droneDone[i] = false;
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

    bool DroneAtTarget(int index)
    {
        return Vector3.Distance(drones[index].transform.position, target.position) <= 2.0f;
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

    // Returns the current state (19 floats) of the given drone.
    float[] GetDroneState(GameObject drone)
    {
        Rigidbody rb = drone.GetComponent<Rigidbody>();
        float[] state = new float[STATE_DIM];
        state[0] = drone.transform.eulerAngles.y;
        state[1] = rb.velocity.magnitude;
        Vector3 toTarget = target.position - drone.transform.position;
        // Use SignedAngle to provide directional context (left vs right)
        state[2] = Vector3.SignedAngle(drone.transform.forward, toTarget, Vector3.up);
        state[3] = toTarget.magnitude;
        
        float minDist1 = float.MaxValue, minDist2 = float.MaxValue;
        float angle1 = 0, angle2 = 0;
        float align1 = 0, align2 = 0; // Alignment (dot product)

        foreach (GameObject other in drones)
        {
            if (other == drone) continue;
            Vector3 toOther = other.transform.position - drone.transform.position;
            float dist = toOther.magnitude;
            float angle = Vector3.SignedAngle(drone.transform.forward, toOther, Vector3.up);
            float alignment = Vector3.Dot(drone.transform.forward, other.transform.forward);

            if (dist < minDist1)
            {
                minDist2 = minDist1;
                angle2 = angle1;
                align2 = align1;

                minDist1 = dist;
                angle1 = angle;
                align1 = alignment;
            }
            else if (dist < minDist2)
            {
                minDist2 = dist;
                angle2 = angle;
                align2 = alignment;
            }
        }
        state[4] = angle1;
        state[5] = minDist1;
        state[6] = angle2;
        state[7] = minDist2;
        
        // Raycasts
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
        
        // New Alignment States
        state[17] = align1;
        state[18] = align2;

        return state;
    }
    
    float CalculateReward(int droneIndex, float[] currentState, int droneFlag, float[] actions)
    {
        float reward = 0f;
        
        float curDist = currentState[3]; // Distance to target
        float prevDist = prevDistToTarget[droneIndex];
        
        // 1. Navigation Reward (Improved)
        // Stronger incentive to move closer
        float distDiff = (prevDist - curDist);
        reward += distDiff * 50f; 
        
        // Continuous penalty for distance to encourage speed
        reward -= curDist * 0.01f;

        // 2. Target Reached / Collision
        if (droneFlag == 1) {
            reward += 300f; // Increased success reward
        }
        else if (droneFlag == 2) {
            reward -= 100f; // Increased collision penalty
        }
        
        // 3. Flocking Behavior
        float minDist1 = currentState[5]; // Nearest neighbor
        float align1 = currentState[17];  // Alignment with nearest

        // Separation (Avoid crowding)
        if (minDist1 < 4f) {
            reward -= (4f - minDist1) * 2f; // Penalty increases as they get closer
        }
        
        // Cohesion (Stay within range, but not too close)
        if (minDist1 >= 5f && minDist1 <= 15f) {
            reward += 0.5f;
        }
        
        // Alignment (Move in same direction)
        if (minDist1 < 20f) {
             reward += align1 * 0.5f; // +0.5 if parallel, -0.5 if opposite
        }
        
        // 4. Obstacle Avoidance
        float minObstacle = 40f;
        for (int i = 8; i <= 16; i++) {
            if (currentState[i] < minObstacle) minObstacle = currentState[i];
        }
        if (minObstacle < 8f) {
            reward -= (8f - minObstacle) * 1.5f;
        }
        
        // 5. Time Penalty (encourage efficiency)
        reward -= 0.1f; 
        
        // 6. Action Smoothness / Anti-Spinning
        if (actions != null)
        {
            float steering = actions[0];
            // Penalize high steering values to prevent spinning
            reward -= Mathf.Abs(steering) * 0.2f;
            
            // Penalize if angular velocity is too high (if we had access, but steering proxy is ok)
        }
        
        // 7. Movement Incentive
        float velocity = currentState[1];
        if (velocity > 1.0f) {
             reward += 0.1f; // Small bonus for moving
        } else {
             reward -= 0.1f; // Penalty for stopping
        }

        return reward;
    }

    // Sends the states of all drones to Python with individual termination flags and rewards
    // Protocol: [State(17), Reward(1), Flag(1)] * NUM_DRONES
    void SendStatesToPython(float[] rewards, int[] flags)
    {
        int packetSize = STATE_DIM + 2; 
        float[] allData = new float[NUM_DRONES * packetSize];
        
        for (int i = 0; i < NUM_DRONES; i++)
        {
            float[] state = GetDroneState(drones[i]);
            Array.Copy(state, 0, allData, i * packetSize, STATE_DIM);
            
            allData[i * packetSize + STATE_DIM] = rewards[i];
            allData[i * packetSize + STATE_DIM + 1] = flags[i];
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
            if (droneDone[i])
                continue;

            // store prev dist for reward calc
            prevDistToTarget[i] = Vector3.Distance(drones[i].transform.position, target.position);
            
            float steering_norm = actions[i * ACTION_DIM + 0]; // [-1,1]
            float throttle_norm = actions[i * ACTION_DIM + 1]; // [-1,1] - Allow backward

            // clip to safe ranges just in case
            steering_norm = Mathf.Clamp(steering_norm, -1f, 1f);
            throttle_norm = Mathf.Clamp(throttle_norm, -1f, 1f); // Allow negative

            ApplyActionToDrone(drones[i], steering_norm, throttle_norm);
        }
    }

    void ApplyActionToDrone(GameObject drone, float steering_norm, float throttle_norm)
    {
        Rigidbody rb = drone.GetComponent<Rigidbody>();
        // Steering: convert [-1,1] -> [-MAX_STEERING_DEG, MAX_STEERING_DEG]
        float targetTurnAngle = steering_norm * MAX_STEERING_DEG;

        // throttle_norm is [-1,1] -> scale to speed/accel
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
        bool allDone = true;

        for (int i = 0; i < NUM_DRONES; i++)
        {
            CollisionDetector detector = drones[i].GetComponent<CollisionDetector>();
            if (detector != null && detector.HasCollided)
            {
                droneDone[i] = true;
            }

            if (!droneDone[i] && DroneAtTarget(i))
            {
                droneDone[i] = true;
            }

            if (!droneDone[i])
            {
                allDone = false;
            }
        }

        if (allDone)
        {
            terminationFlag = 1;
        }
        else
        {
            terminationFlag = 0;
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
            if (actions == null) yield break; // disconnected

            // Special reset signal from Python (first value -99)
            if (actions.Length > 0 && Mathf.Approximately(actions[0], -99f))
            {
                ResetDrones();
                SendStatesToPython(new float[NUM_DRONES], new int[NUM_DRONES]);
                continue;
            }

            // 2. Apply actions to drones (and store prev dists)
            ApplyActionsToDrones(actions);

            // 3. Wait for physics update
            yield return new WaitForFixedUpdate();

            // 4. Check termination conditions AFTER physics update
            CheckTerminationConditions();
            
            // 5. Calculate Rewards based on new state
            float[] rewards = new float[NUM_DRONES];
            int[] droneFlags = new int[NUM_DRONES];
            for(int i=0; i<NUM_DRONES; i++) {
                float[] state = GetDroneState(drones[i]);
                
                // Extract action for this drone to pass to reward function
                float[] droneAction = new float[ACTION_DIM];
                if (actions != null)
                    Array.Copy(actions, i * ACTION_DIM, droneAction, 0, ACTION_DIM);
                else 
                    droneAction = null;

                int flag = 0;
                CollisionDetector detector = drones[i].GetComponent<CollisionDetector>();
                bool collided = detector != null && detector.HasCollided;
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

                droneFlags[i] = flag;
                rewards[i] = CalculateReward(i, state, flag, droneAction);
            }
            
            // 6. Send the NEXT state, Reward, and Flag
            SendStatesToPython(rewards, droneFlags);

            // 7. If episode ended, reset and send initial state for next episode
            if (terminationFlag != 0)
            {
                ResetDrones();
                // Send new initial state (Rewards=0)
                SendStatesToPython(new float[NUM_DRONES], new int[NUM_DRONES]);
            }
        }
    }

    void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
    }
}

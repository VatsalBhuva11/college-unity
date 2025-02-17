using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class FlockingDrones : MonoBehaviour
{
    public GameObject[] drones; // Assign 3 drones in Unity Inspector
    public Transform target;
    private TcpClient client;
    private NetworkStream stream;
    private const int STATE_DIM = 17;
    private const int ACTION_DIM = 2;
    private const int NUM_DRONES = 3;
    private const float TARGET_THRESHOLD = 2.0f; // Distance threshold to consider target reached
    private bool episodeEnded = false;
    private Vector3[] initialDronePositions; // Array to store initial positions of drones
    private Quaternion[] initialDroneRotations; // Array to store initial rotations of drones

    void Start()
    {
        ConnectToPython();
        StoreInitialPositions(); // Store initial positions and rotations
    }

    void ResetDrones()
    {
        for (int i = 0; i < NUM_DRONES; i++)
        {
            Rigidbody rb = drones[i].GetComponent<Rigidbody>();
            rb.velocity = Vector3.zero; // Reset velocity
            rb.angularVelocity = Vector3.zero; // Reset angular velocity
            drones[i].transform.position = initialDronePositions[i]; // Reset position
            drones[i].transform.rotation = initialDroneRotations[i]; // Reset rotation
            drones[i].transform.SetPositionAndRotation(initialDronePositions[i], initialDroneRotations[i]);

            // Reset collision flag (if applicable)
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

    void FixedUpdate()
    {
        if (episodeEnded)
        {
            ResetDrones(); // Reset drones to initial positions
            episodeEnded = false; // Reset the episode flag
            return;
        }

        foreach (GameObject drone in drones)
        {
            float[] state = GetDroneState(drone);
            SendStateToPython(state);
            float[] action = ReceiveActionFromPython();
            ApplyActionToDrone(drone, action);
        }

        // Check for episode termination conditions
        if (CheckCollision() || CheckTargetReached())
        {
            episodeEnded = true;
            Debug.Log("Episode Ended");
            SendTerminationSignal();
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

    float[] GetDroneState(GameObject drone)
    {
        Rigidbody rb = drone.GetComponent<Rigidbody>();
        float[] state = new float[STATE_DIM];

        // 1. Drone's angle (yaw in degrees)
        state[0] = drone.transform.eulerAngles.y;

        // 2. Drone's velocity (magnitude of velocity vector)
        state[1] = rb.velocity.magnitude;

        // 3-4. Angle and distance to target
        Vector3 toTarget = target.position - drone.transform.position;
        state[2] = Vector3.Angle(drone.transform.forward, toTarget);
        state[3] = toTarget.magnitude;

        // 5-8. Angle and distance to nearest two neighbors
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

        // 9-17. Nine virtual range finders (raycasts for obstacles in different directions)
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
                state[8 + i] = 40f; // Max range if no obstacle detected
            }
        }

        foreach (float x in state)
        {
            Debug.Log(x + " ");
        }


        return state;
    }

    void SendStateToPython(float[] state)
    {
        byte[] data = new byte[STATE_DIM * 4];
        Buffer.BlockCopy(state, 0, data, 0, data.Length);
        stream.Write(data, 0, data.Length);
        stream.Flush();
    }

    float[] ReceiveActionFromPython()
    {
        byte[] data = new byte[ACTION_DIM * 4];

        if (stream.DataAvailable)  // Only read if data is available
        {
            stream.Read(data, 0, data.Length);
            float[] action = new float[ACTION_DIM];
            Buffer.BlockCopy(data, 0, action, 0, data.Length);
            return action;
        }
        else
        {
            return new float[] { 0, 0 };  // Default action (no movement) to avoid blocking
        }
    }

    void ApplyActionToDrone(GameObject drone, float[] action)
    {
        Debug.Log($"Received action: {action[0]}, {action[1]}");

        Rigidbody rb = drone.GetComponent<Rigidbody>();

        float steering = action[0];  // a1: steering control
        float throttle = action[1];  // a2: throttle control

        // Apply steering (rotation)
        float maxSteeringAngle = Mathf.PI / 4; // ±45 degrees
        float turnAngle = steering * maxSteeringAngle * Mathf.Rad2Deg;
        drone.transform.Rotate(0, turnAngle * Time.fixedDeltaTime, 0);

        // Apply throttle (forward acceleration)
        Vector3 force = drone.transform.forward * throttle * 10f; // Adjust force multiplier as needed
        rb.AddForce(force, ForceMode.Acceleration);
    }

    bool CheckCollision()
    {
        foreach (GameObject drone in drones)
        {
            // Check if any drone has collided with an obstacle
            if (drone.GetComponent<CollisionDetector>().HasCollided)
            {
                return true;
            }
        }
        return false;
    }

    bool CheckTargetReached()
    {
        foreach (GameObject drone in drones)
        {
            // Check if all drones are within the target threshold
            if (Vector3.Distance(drone.transform.position, target.position) > TARGET_THRESHOLD)
            {
                return false;
            }
        }
        return true;
    }

    void SendTerminationSignal()
    {
        // Send a special signal to Python to indicate episode termination
        byte[] signal = new byte[1] { 0xFF }; // Example termination signal
        stream.Write(signal, 0, signal.Length);
        stream.Flush();
    }

    void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
    }
}
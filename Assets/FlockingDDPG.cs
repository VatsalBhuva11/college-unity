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

    void Start()
    {
        ConnectToPython();
    }

    void FixedUpdate()
    {
        foreach (GameObject drone in drones)
        {
            float[] state = GetDroneState(drone);
            SendStateToPython(state);
            float[] action = ReceiveActionFromPython();
            ApplyActionToDrone(drone, action);
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
            if (Physics.Raycast(drone.transform.position, direction, out hit, 50f))
            {
                state[8 + i] = hit.distance;
            }
            else
            {
                state[8 + i] = 50f; // Max range if no obstacle detected
            }
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
        stream.Read(data, 0, data.Length);
        float[] action = new float[ACTION_DIM];
        Buffer.BlockCopy(data, 0, action, 0, data.Length);
        return action;
    }

    void ApplyActionToDrone(GameObject drone, float[] action)
    {
        Rigidbody rb = drone.GetComponent<Rigidbody>();
        Vector3 force = new Vector3(action[0], 0, action[1]); // Assuming actions are (x, z) directional forces
        rb.AddForce(force, ForceMode.Acceleration);
    }

    void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
    }
}
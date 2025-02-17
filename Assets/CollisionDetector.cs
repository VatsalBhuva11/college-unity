using UnityEngine;

public class CollisionDetector : MonoBehaviour
{
    public bool HasCollided = false;

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Obstacle"))
        {
            HasCollided = true;
        }
    }
}
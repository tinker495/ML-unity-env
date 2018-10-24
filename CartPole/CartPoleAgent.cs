using System;
using UnityEngine;
using System.Linq;
using MLAgents;

public class CartPoleAgent : Agent
{
    Rigidbody rBody;
    public override void InitializeAgent()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Axis;
    public Transform Pole;
    public Transform Goal;
    public Rigidbody PoleRigdbody;

    public override void AgentReset()
    {
        this.transform.position = Axis.position + new Vector3(UnityEngine.Random.value * 1.0f - 0.5f, 0, 0);
        Goal.position = Axis.position + new Vector3(UnityEngine.Random.value * 7.0f - 3.5f, 0, -0.1f);
        this.rBody.velocity = Vector3.zero;

        Pole.position = this.transform.position + new Vector3(0,0.75f,0);
        Pole.eulerAngles = new Vector3(0, 0, UnityEngine.Random.value * 20.0f - 10.1f);
        PoleRigdbody.angularVelocity = Vector3.zero;
        PoleRigdbody.velocity = Vector3.zero;
    }

    public override void CollectObservations()
    {

        float CartPos = Axis.position.x - this.transform.position.x;
        float GoalError = Goal.position.x - this.transform.position.x;
        float PoleAng = Pole.eulerAngles.z;
        if (PoleAng > 180.0f)
        {
            PoleAng = PoleAng - 360.0f;
        }
        //Vector3 relativePosition = this.transform.position; // - this.transform.position;

        AddVectorObs(CartPos);
        AddVectorObs(GoalError);
        AddVectorObs(this.rBody.velocity.x);
        AddVectorObs(PoleAng);
        AddVectorObs(PoleRigdbody.angularVelocity.z);
    }

    public float speed = 30.0f;

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        // Rewards
        float CartPos = Axis.position.x - this.transform.position.x;
        float GoalError = Goal.position.x - this.transform.position.x;
        SetReward(1.0f + 1 / (1 + 20f*Mathf.Abs(GoalError)));

        if ((Pole.eulerAngles.z > 40 && Pole.eulerAngles.z < 360 - 40)||(CartPos > 5.0f||CartPos < -5.0f))
        {
            Done();
            SetReward(-1.0f);
        }
        // Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = Mathf.Clamp(vectorAction[0], -1.0f, 1.0f);
        rBody.AddForce(speed * controlSignal);
    }
}

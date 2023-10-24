import torch
import math
def Ackermann(
        lin_vel: torch.Tensor, 
        ang_vel: torch.Tensor, 
        device: torch.device,
        wl = 0.849,
        d_fr = 0.77,
        d_mw = 0.894,
):
    """ 
    Ackermann steering model for the rover
    wl = wheelbase length
    d_fr = distance between front and rear wheels
    d_mw = distance between middle wheels 
    """

    # Checking the direction of the linear and angular velocities
    direction: torch.Tensor = torch.sign(lin_vel)
    turn_direction: torch.Tensor = torch.sign(ang_vel)

    # Taking the absolute values of the velocities
    lin_vel = torch.abs(lin_vel)
    ang_vel = torch.abs(ang_vel)

    # Calculates the turning radius of the rover, returns inf if ang_vel is 0
    not_zero_condition = torch.logical_not(ang_vel == 0) & torch.logical_not(lin_vel == 0)
    
    minimum_radius = (d_mw / 2) + 0.05 # 5 extra cm make operation more smooth
    turning_radius: torch.Tensor = torch.where(not_zero_condition, lin_vel / ang_vel, torch.tensor(float('inf'), device=device))
    print(turning_radius)
    turning_radius = torch.where(turning_radius < minimum_radius, minimum_radius, turning_radius)
    
    # Calculating the turning radius of the front wheels
    r_ML = turning_radius - (d_mw / 2)
    r_MR = turning_radius + (d_mw / 2)
    r_FL = turning_radius - (d_fr / 2)
    r_FR = turning_radius + (d_fr / 2)
    r_RL = turning_radius - (d_fr / 2)
    r_RR = turning_radius + (d_fr / 2)

    # Steering angles

    wl = torch.ones_like(r_FL) * wl # Repeat wl as tensor
    #print(turning_radius)
    theta_FL = torch.atan2(wl, r_FL) * direction * turn_direction
    theta_FR = torch.atan2(wl, r_FR) * direction * turn_direction
    theta_RL = -torch.atan2(wl, r_RL) * direction * turn_direction
    theta_RR = -torch.atan2(wl, r_RR) * direction * turn_direction

    # Wheel velocities (m/s) 
    # if ang_vel is 0, wheel velocity is equal to linear velocity
    vel_FL = torch.where(ang_vel == 0, lin_vel, (r_FL * ang_vel)) * direction
    vel_FR = torch.where(ang_vel == 0, lin_vel, (r_FR * ang_vel)) * direction
    vel_RL = torch.where(ang_vel == 0, lin_vel, (r_RL * ang_vel)) * direction
    vel_RR = torch.where(ang_vel == 0, lin_vel, (r_RR * ang_vel)) * direction
    vel_ML = torch.where(ang_vel == 0, lin_vel, (r_ML * ang_vel)) * direction
    vel_MR = torch.where(ang_vel == 0, lin_vel, (r_MR * ang_vel)) * direction

    # Stack the wheel velocities and steering angles
    wheel_velocities = torch.stack([vel_FL, vel_FR, vel_RL, vel_RR, vel_ML, vel_MR], dim=1)
    steering_angles = torch.stack([theta_FL, theta_FR, theta_RL, theta_RR], dim=1)
    
    return torch.cat([steering_angles, wheel_velocities], dim=1)


if __name__ == "__main__":
    lin = torch.ones((2,1)) * 1
    ang = torch.ones((2,1)) * 0
    #print(lin)
    a = Ackermann(lin,ang, 'cpu')
    print(a[1,0:10])

    #print(a.shape)

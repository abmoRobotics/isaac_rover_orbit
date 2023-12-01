import torch


def tensor_quat_to_eul(quats, device):
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    # Quaternions format: W, X, Y, Z
    # Quat index:         0, 1, 2, 3
    # Euler angles:       ZYX

    euler_angles = torch.zeros([len(quats), 3], device=device)
    ones = torch.ones([len(quats)], device=device)
    zeros = torch.zeros([len(quats)], device=device)

    #Roll
    sinr_cosp = 2 * (quats[:,0] * quats[:,1] + quats[:,2] * quats[:,3])
    cosr_cosp = ones - (2 * (quats[:,1] * quats[:,1] + quats[:,2] * quats[:,2]))
    euler_angles[:,0] = torch.atan2(sinr_cosp, cosr_cosp)

    #Pitch
    sinp = 2 * (quats[:,0]*quats[:,2] - quats[:,3] * quats[:,1])
    condition = (torch.sign(sinp - ones) >= zeros)
    euler_angles[:,1] = torch.where(condition, torch.copysign((ones*torch.pi)/2, sinp), torch.asin(sinp))

    #Yaw
    siny_cosp = 2 * (quats[:,0] * quats[:,3] + quats[:,1] * quats[:,2])
    cosy_cosp = ones - (2 * (quats[:,2] * quats[:,2] + quats[:,3] * quats[:,3]))
    euler_angles[:,2] = torch.atan2(siny_cosp, cosy_cosp)

    return euler_angles

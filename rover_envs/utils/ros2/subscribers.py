from geometry_msgs.msg import Twist
from rclpy.node import Node


class TwistSubscriber(Node):

    def __init__(self):
        super().__init__('twist_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.listener_callback,
            10)
        self.subscription
        self.velocity = 0.0
        self.angular = 0.0

    def listener_callback(self, msg):
        self.velocity = msg.linear.x
        self.angular = msg.angular.z
        # self.get_logger().info('I heard: "%s"' % msg.linear.x)

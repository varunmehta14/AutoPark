# Autonomous Path Planning and Parking Simulation  

## **Overview**  
This project explores **autonomous parking** using **path planning, obstacle detection, and motion tracking** techniques. It implements **multi-vehicle coordination** in **2D and 3D environments**, initially using **CARLA** but later optimized for lightweight **Pygame-based simulations**. The simulation provides a scalable, efficient testing framework for **autonomous vehicle navigation** in parking lots.  

## **Technologies Used**  

### **Path Planning**  
- **A* Algorithm**: Computes optimal paths while avoiding obstacles.  
- **Graph-Based Cost Calculation**: Utilizes heuristic and traversal cost functions to generate smooth parking trajectories.  
- **Cubic Spline Interpolation**: Applied to refine path smoothness, ensuring realistic vehicle motion.  

### **Motion Control & Vehicle Dynamics**  
- **Pure Pursuit Algorithm**: Implements smooth waypoint tracking, replacing the more complex **Kinematic Bicycle Model**.  
- **PID Controller**: Regulates speed and braking to ensure stable acceleration and deceleration.  

### **Sensor Simulation & Obstacle Detection**  
- **LIDAR-Based Raycasting**: Simulates real-time environment scanning, detecting obstacles dynamically.  
- **Real-Time Collision Avoidance**: The system continuously adjusts paths to prevent collisions with static and dynamic objects.  

### **Simulation Frameworks**  
- **CARLA**: Used initially for high-fidelity simulation and real-world physics integration.  
- **Pygame**: Provides a lightweight 2D simulation environment, allowing for rapid iteration and scalability.  

### **Multi-Agent Coordination**  
- **Dynamic Replanning**: Vehicles adjust their routes in response to environmental changes, ensuring efficient parking.  
- **Multi-Vehicle Navigation**: Implements coordination strategies to avoid conflicts and optimize parking lot usage.  

## **Results & Performance**  
- **Optimized A* algorithm** reduces path computation time to **<0.5s in 2D** and **~0.8s in 3D**.  
- **Obstacle detection success rate**: **99% for static obstacles, 92% for dynamic obstacles**.  
- **Multi-agent simulation** enables real-time adjustments, improving coordination efficiency.  

## **Future Enhancements**  
- **Integration with Reinforcement Learning (RL)** for adaptive decision-making.  
- **Improved 3D motion dynamics** with a refined vehicle model.  
- **Parallelized path planning** to enhance real-time response in complex environments.  

## **Contributors**  
- **Varun Mehta** – Path planning, LIDAR-based obstacle detection, multi-agent navigation.  
- **Niveditha Madegowda** – A* implementation, vehicle control optimization, system architecture.  
- **Smrithi Agrawal** – Dynamic rerouting, PID controller tuning, multi-vehicle interaction.  
- **Rutuja Pote** – Pure Pursuit algorithm, collision avoidance, vehicle coordination.  
- **Shreyas Agrawal** – 2D multi-agent system, simulation environment setup.  

---

This project provides a **scalable, computationally efficient** approach to **autonomous parking**, balancing **realism and performance** for future **self-driving applications**.

## 🎨 View Presentation  

[![View on Canva](https://img.shields.io/badge/View%20on-Canva-blue?style=for-the-badge&logo=canva)](https://www.canva.com/design/DAEZw--q3Do/9kYITxLAQyoL_ypAXR4yTg/view?utm_content=DAEZw--q3Do&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h0aaebf22f9)

![final_op](https://github.com/user-attachments/assets/887e2ac3-b45e-453e-81ef-0b488a447687)



https://github.com/user-attachments/assets/a5025e6f-2cbe-43a6-ac6e-8d58dffcd8a2



https://github.com/user-attachments/assets/51abf4ad-2c89-4b57-a336-89ca063453ad



https://github.com/user-attachments/assets/a3b1794e-9e4f-444a-a6ed-58ff1b771bd4





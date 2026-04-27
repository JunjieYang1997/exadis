/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  This module implements cross-slip for FCC crystals in parallel fashion.
 *  It is the parallelization of file cross_slip_serial.cpp
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CROSS_SLIP_PARALLEL_H
#define EXADIS_CROSS_SLIP_PARALLEL_H

#include "force.h"
#include "cross_slip_serial.h"
#include "topology_parallel.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        CrossSlipParallel
 *
 *-------------------------------------------------------------------------*/
template<class F>
class CrossSlipParallel : public CrossSlip {
private:
    F* force;//定义一个指向F类型的指针，F是一个模板参数，表示力的类型
    
public:
    CrossSlipParallel(System* system, Force* _force)
    {
        // Check and assign force kernel
        force = dynamic_cast<F*>(_force);
        if (force == nullptr)
            ExaDiS_fatal("Error: inconsistent force type in CrossSlipParallel\n");
    }//构造函数，接受一个System指针和一个Force指针作为参数，并将Force指针转换为F类型的指针，如果转换失败则抛出错误
    
    /*-----------------------------------------------------------------------
     *    Struct:     CrossSlipEvent
     *                Structure to hold cross-slip event information.
     *---------------------------------------------------------------------*/
    struct CrossSlipEvent {
        int type; 
        Vec3 p0, p1;
        Vec3 plane;
    };//定义一个结构体CrossSlipEvent，用于存储交滑移事件的信息，包括事件类型、两个位置向量和一个平面向量
    
    /*-----------------------------------------------------------------------
     *    Struct:     FindCrossSlipEvents
     *                Structure that implements the kernel to determine the
     *                cross-slip events that must be executed. Each node
     *                is assigned a team of threads for parallel force
     *                calculations.
     *---------------------------------------------------------------------*/
    struct FindCrossSlipEvents
    {
        System* system;
        DeviceDisNet* net;
        F* force;
        NeighborList* neilist;
        
        double eps, thetacrit, sthetacrit, s2thetacrit;//容差值，临界角度的正弦值和正弦值的平方
        double shearModulus, areamin;//剪切模量和最小面积
        double noiseFactor, weightFactor;//噪声因子和权重因子
        
        Mat33 R, Rinv;//晶体方向矩阵和其逆矩阵
        
        Kokkos::View<int*, T_memory_shared> count;//一个共享内存的Kokkos视图，用于计数交滑移事件的数量
        Kokkos::View<int*, T_memory_space> csnodes;//一个Kokkos视图，用于存储需要进行交滑移的节点的索引
        Kokkos::View<CrossSlipEvent*, T_memory_shared> events;//一个共享内存的Kokkos视图，用于存储交滑移事件的信息
        
        FindCrossSlipEvents(System* _system, DeviceDisNet* _net, F* _force) :
        system(_system), net(_net), force(_force)
        {
            eps = 1e-6;
            if (system->crystal.type == FCC_CRYSTAL)
                thetacrit = 2.0 / 180.0 * M_PI;//对于FCC晶体，临界角度设为2度
            else if (system->crystal.type == BCC_CRYSTAL)
                thetacrit = 0.5 / 180.0 * M_PI;//对于BCC晶体，临界角度设为0.5度
            sthetacrit = sin(thetacrit);
            s2thetacrit = sthetacrit * sthetacrit;
            shearModulus = system->params.MU;
            
            areamin = 2.0 * system->params.rtol * system->params.maxseg;
            areamin = MIN(areamin, system->params.minseg * system->params.minseg * sqrt(3.0) / 4.0);
            
            noiseFactor = 1e-5;
            weightFactor = 1.0;
            
            R = system->crystal.R;
            Rinv = system->crystal.Rinv;
        }//构造函数，接受一个System指针、一个DeviceDisNet指针和一个F类型的指针作为参数，并初始化一些成员变量，包括容差值、临界角度的正弦值、剪切模量、最小面积、噪声因子和权重因子，以及晶体方向矩阵和其逆矩阵
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i) const
        {
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            auto conn = net->get_conn();
            auto cell = net->cell;//获取网络中的节点、线段、连接信息和单元信息
            
            if (conn[i].num != 2) return;//如果连接数不等于2，说明该节点不是一个交滑移事件的候选节点，直接返回
            if (nodes[i].constraint != UNCONSTRAINED) return;//如果节点受到约束，说明它不能进行交滑移，直接返回
            
            int s = conn[i].seg[0];//获取连接的第一个线段的索引
            Vec3 burg = segs[s].burg;//获取该线段的Burgers向量
            double burgSize = burg.norm();
            burg = burg.normalized();
            
            Vec3 burgCrystal = Rinv * burg;
            burgCrystal = burgCrystal.normalized();
            
            if (system->crystal.type == FCC_CRYSTAL) {
            
                // Only consider glide dislocations. If the Burgers vector is not
                // a [1 1 0] type, ignore it.
                if (!((fabs(fabs(burgCrystal.x)-fabs(burgCrystal.y)) < eps) &&
                      (fabs(burgCrystal.z) < eps)) &&
                    !((fabs(fabs(burgCrystal.y)-fabs(burgCrystal.z)) < eps) &&
                      (fabs(burgCrystal.x) < eps )) &&
                    !((fabs(fabs(burgCrystal.z)-fabs(burgCrystal.x)) < eps) &&
                      (fabs(burgCrystal.y) < eps))) {
                    return;//如果Burgers向量在晶体坐标系下不满足[1 1 0]类型的条件，说明它不是一个滑移位错，直接返回
                }

                if ((fabs(burgCrystal.x) < eps) && (fabs(burgCrystal.y) < eps) &&
                    (fabs(burgCrystal.z) < eps)) {
                    return;//如果Burgers向量在晶体坐标系下的分量都接近于零，说明它不是一个有效的位错，直接返回
                }
                
                // Also test that the segment resides on a (1 1 1) plane, since these
                // are the only planes where cross-slip occurs.
                Vec3 plane = segs[s].plane.normalized();
                Vec3 planeCrystal = Rinv * plane;
                if ((fabs(fabs(planeCrystal.x) - fabs(planeCrystal.y)) > eps) ||
                    (fabs(fabs(planeCrystal.y) - fabs(planeCrystal.z)) > eps)) {
                    return; // not a {111} plane
                }
                
            } else if (system->crystal.type == BCC_CRYSTAL) {
                
                // Only consider <111> dislocations
                if (fabs(burgCrystal.x * burgCrystal.y * burgCrystal.z) < eps) {
                    return;//如果Burgers向量在晶体坐标系下的分量乘积接近于零，说明它不是一个<111>类型的位错，直接返回
                }
                
            }
            
            int n1 = conn[i].node[0];
            int n2 = conn[i].node[1];
            
            Vec3 nodep = nodes[i].pos;
            Vec3 nbr1p = cell.pbc_position(nodep, nodes[n1].pos);
            Vec3 nbr2p = cell.pbc_position(nodep, nodes[n2].pos);
            
            // If the node is a point on a long screw then we can consider
            // it for possible cross slip.
            Vec3 vec1 = nbr1p - nbr2p;
            Vec3 vec2 = nodep - nbr1p;
            Vec3 vec3 = nodep - nbr2p;//计算两个邻居节点之间的向量以及当前节点与两个邻居节点之间的向量
            
            // Calculate some test conditions
            double test1 = dot(vec1, burg);
            double test2 = dot(vec2, burg);
            double test3 = dot(vec3, burg);
            
            test1 = test1 * test1;
            test2 = test2 * test2;
            test3 = test3 * test3;

            double testmax1 = dot(vec1, vec1);
            double testmax2 = dot(vec2, vec2);
            double testmax3 = dot(vec3, vec3);//计算上述向量与Burgers向量的点积的平方，以及上述向量的长度的平方
            
            // Set up the tests to see if this dislocation is close enough to
            // screw to be considered for cross slip.  For a segment to be close
            //to screw it must be within 2*thetacrit defined above
            bool seg1_is_screw = ((testmax2 - test2) < (testmax2 * s2thetacrit));
            bool seg2_is_screw = ((testmax3 - test3) < (testmax3 * s2thetacrit));
            bool bothseg_are_screw =
                (((testmax2 - test2) < (4.0 * testmax2 * s2thetacrit)) &&
                 ((testmax3 - test3) < (4.0 * testmax3 * s2thetacrit)) &&
                 ((testmax1 - test1) < (testmax1 * s2thetacrit)));//根据上述计算的条件，判断两个线段是否接近于螺旋位错，如果满足条件则将该节点的索引存储在csnodes视图中，以便后续处理交滑移事件
                 
            if (seg1_is_screw || seg2_is_screw || bothseg_are_screw) {
                int idx = Kokkos::atomic_fetch_add(&count(0), 1);
                csnodes(idx) = i;
            }//如果任一线段接近于螺旋位错或者两个线段都接近于螺旋位错，则使用原子操作将该节点的索引添加到csnodes视图中，并增加计数器的值
        }
        
        KOKKOS_INLINE_FUNCTION//定义一个内联函数operator()，接受一个整数i作为参数，表示节点的索引，在该函数中实现了判断交滑移事件的逻辑，并将满足条件的节点索引存储在csnodes视图中，同时还计算了相关的物理量和条件，以便后续处理交滑移事件
        void operator() (const team_handle& team) const
        {
            int tid = team.team_rank();
            int lid = team.league_rank();
            int i = csnodes(lid); // node id
            
            // Flag no event type
            if (tid == 0) events(lid).type = -1;
            
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            auto conn = net->get_conn();
            auto cell = net->cell;//获取网络中的节点、线段、连接信息和单元信息
            
            // Recompute some info about the local node
            int s = conn[i].seg[0];
            Vec3 burg = segs[s].burg;
            double burgSize = burg.norm();
            burg = burg.normalized();
            
            Vec3 burgCrystal = Rinv * burg;
            burgCrystal = burgCrystal.normalized();
            Vec3 plane = segs[s].plane.normalized();
            Vec3 planeCrystal = Rinv * plane;
            
            int n1 = conn[i].node[0];
            int n2 = conn[i].node[1];
            
            Vec3 nodep = nodes[i].pos;
            Vec3 nbr1p = cell.pbc_position(nodep, nodes[n1].pos);
            Vec3 nbr2p = cell.pbc_position(nodep, nodes[n2].pos);
            
            Vec3 vec1 = nbr1p - nbr2p;
            Vec3 vec2 = nodep - nbr1p;
            Vec3 vec3 = nodep - nbr2p;
            
            double test1 = dot(vec1, burg);
            double test2 = dot(vec2, burg);
            double test3 = dot(vec3, burg);
            
            test1 = test1 * test1;
            test2 = test2 * test2;
            test3 = test3 * test3;

            double testmax1 = dot(vec1, vec1);
            double testmax2 = dot(vec2, vec2);
            double testmax3 = dot(vec3, vec3);
            
            bool seg1_is_screw = ((testmax2 - test2) < (testmax2 * s2thetacrit));
            bool seg2_is_screw = ((testmax3 - test3) < (testmax3 * s2thetacrit));
            bool bothseg_are_screw =
                (((testmax2 - test2) < (4.0 * testmax2 * s2thetacrit)) &&
                 ((testmax3 - test3) < (4.0 * testmax3 * s2thetacrit)) &&
                 ((testmax1 - test1) < (testmax1 * s2thetacrit)));//重新计算一些关于当前节点的信息，包括Burgers向量、线段所在的平面、邻居节点的位置以及相关的物理量和条件，以便后续处理交滑移事件
            
            // Since we will likely need to locally modify the network
            // let's first create a new configuration within a temporary, 
            // local SplitNet instance (from Topology)
            SplitDisNet splitnet(net, neilist);//分离网络实例，用于在本地修改网络结构以评估交滑移事件的影响
            
            // Compute the nodal force (initially in laboratory frame)
            Vec3 fLab = force->node_force(system, &splitnet, i, team);//计算当前节点的力，初始情况下是在实验室坐标系下计算的
            
            // Set the force threshold for noise level within the code
            double L1 = sqrt(testmax2);
            double L2 = sqrt(testmax3);
            double fnodeThreshold = noiseFactor * shearModulus * burgSize * 
                                    0.5 * (L1 + L2);
            
            Mat33 glideDirCrystal = Mat33().zero();
            int numGlideDir = 0; // Number of cross-slip glide directions 这个0是用来初始化numGlideDir的，表示交滑移滑移方向的数量，后续根据晶体类型进行设置
            
            if (system->crystal.type == FCC_CRYSTAL) {
                // Find which glide planes the segments are on
                // e.g. for burg = [ 1  1  0 ], the two glide directions are
                //                 [ 1 -1  2 ] and
                //                 [ 1 -1 -2 ]
                // Use Burgers vectors in crystal frame to generate initial glide
                // planes in crystal frame.
                numGlideDir = 2;//对于FCC晶体，交滑移的滑移方向有两个，分别是[1 -1 2]和[1 -1 -2]，根据Burgers向量在晶体坐标系下的分量来生成初始的滑移平面在晶体坐标系下的表示
                double tmp = 1.0;
                for (int j = 0; j < 3; j++) {
                    if (fabs(burgCrystal[j]) > eps) {
                        glideDirCrystal[0][j] = (burgCrystal[j]*tmp > 0) ? 1.0 : -1.0;
                        glideDirCrystal[1][j] = (burgCrystal[j]*tmp > 0) ? 1.0 : -1.0;
                        tmp = -1.0;
                    } else {
                        glideDirCrystal[0][j] =  2.0;
                        glideDirCrystal[1][j] = -2.0;
                    }//根据Burgers向量在晶体坐标系下的分量，设置两个滑移方向在晶体坐标系下的表示，如果分量大于容差值，则对应的滑移方向分量为1或-1，否则为2或-2，具体取决于Burgers向量分量与一个临时变量的乘积的符号
                }
                
                // Normalization
                glideDirCrystal[0] = sqrt(1.0/6.0) * glideDirCrystal[0];//对两个滑移方向进行归一化，使其长度为sqrt(1/6)，这是因为对于FCC晶体，交滑移的滑移方向是[1 -1 2]和[1 -1 -2]，它们的长度为sqrt(6)，所以需要除以sqrt(6)来进行归一化
                glideDirCrystal[1] = sqrt(1.0/6.0) * glideDirCrystal[1];//同上，对第二个滑移方向进行归一化
                
            } else if (system->crystal.type == BCC_CRYSTAL) {
                // Find which glide planes the segments are on. Initial
                // glidedir array contains glide directions in crystal frame
                // For BCC geometry burgCrystal should be of <1 1 1> type
                numGlideDir = 3;//对于BCC晶体，交滑移的滑移方向有三个，根据Burgers向量在晶体坐标系下的分量来生成初始的滑移平面在晶体坐标系下的表示
                Mat33 tmp33 = outer(burgCrystal, burgCrystal);
                for (int m = 0; m < 3; m++)
                    for (int n = 0; n < 3; n++)
                        glideDirCrystal[m][n] = ((m==n)-tmp33[m][n]) * sqrt(1.5);

                // glideDirCrystal should now contain the three <112> type
                // directions that a screw dislocation may move in if glide
                // is restricted to <110> type glide planes
            }
            
            int s1 = conn[i].seg[0];
            int s2 = conn[i].seg[1];
            Vec3 segplane1 = segs[s1].plane;
            Vec3 segplane2 = segs[s2].plane;//获取当前节点连接的两个线段所在的平面
            
            // Rotations
            Mat33 glideDirLab;
            for (int j = 0; j < 3; j++)
                glideDirLab[j] = R * glideDirCrystal[j];
            segplane1 = Rinv * segplane1;
            segplane2 = Rinv * segplane2;
            Vec3 fCrystal = Rinv * fLab;
            
            Vec3 tmp3  = glideDirCrystal * segplane1;
            Vec3 tmp3B = glideDirCrystal * segplane2;
            Vec3 tmp3C = glideDirCrystal * fCrystal;//将滑移方向从晶体坐标系转换到实验室坐标系，并计算滑移方向与两个线段所在平面的点积，以及滑移方向与力的点积，以便后续判断交滑移事件的条件
            
            // For FCC there are only two slip planes for screw dislocation
            int plane1 = 0;
            int plane2 = 0;
            int fplane = 0;
            
            for (int j = 1; j < numGlideDir; j++) {
                plane1 = (fabs(tmp3[j])  < fabs(tmp3[plane1]) ) ? j : plane1;
                plane2 = (fabs(tmp3B[j]) < fabs(tmp3B[plane2])) ? j : plane2;
                fplane = (fabs(tmp3C[j]) > fabs(tmp3C[fplane])) ? j : fplane;
            }//对于FCC晶体，交滑移的滑移方向只有两个，所以直接比较两个滑移方向与线段所在平面的点积的绝对值，找到最小的那个对应的平面索引；对于力的点积，则找到最大的那个对应的平面索引
            
            // Calculate the new plane in the lab frame
            Vec3 newplane = cross(burg, glideDirLab[fplane]).normalized();//根据Burgers向量和选定的滑移方向计算新的平面在实验室坐标系下的表示
            
            if (bothseg_are_screw && (plane1 == plane2) && (plane1 != fplane) &&
                (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane1])+fnodeThreshold))) {
                
                // Both segments are close to screw and the average direction
                // is close to screw.
                
                // Determine if the neighbor nodes should be considered immobile
                bool pinned1 = node_pinned(system, net, n1, plane1, glideDirLab, numGlideDir);
                bool pinned2 = node_pinned(system, net, n2, plane2, glideDirLab, numGlideDir);//如果两个线段都接近于螺旋位错，并且它们所在的平面相同但不等于选定的滑移方向所在的平面，并且力在选定的滑移方向上的分量大于一个阈值，则满足交滑移事件的条件，此时需要判断两个邻居节点是否被固定在原位，如果被固定则不能进行交滑移操作，否则可以进行交滑移操作
                
                if (pinned1) {
                    if ((!pinned2) || ((testmax1-test1) < (eps*eps*burgSize*burgSize))) {
                        
                        double vec1dotb = dot(vec1, burg);
                        double vec2dotb = dot(vec2, burg);//如果邻居节点1被固定，但邻居节点2没有被固定或者邻居节点1与Burgers向量的点积的平方接近于线段长度的平方，则可以进行交滑移操作，此时需要计算新的节点位置，并评估在新的配置下节点上的力，以确定是否继续进行交滑移事件
                        
                        if (!pinned2) {
                            // Neighbor 2 can be moved, so proceed with the
                            // cross-slip operation.
                            nbr2p = nbr1p - vec1dotb * burg;
                        }//如果邻居节点2没有被固定，则可以直接计算新的位置；如果邻居节点2被固定，则说明它已经在螺旋方向上与邻居节点1完全对齐，所以不需要移动它
                        
                        // If neighbor2 is pinned, it is already perfectly
                        // aligned with neighbor1 in the screw direction
                        // so there is no need to move it.
                        nodep = nbr1p + vec2dotb * burg;
                        
                        double fdotglide = dot(fLab, glideDirLab[fplane]);
                        double tmp = areamin / fabs(vec1dotb) * 2.0 * (1.0 + eps) * SIGN(fdotglide);
                        nodep += tmp * glideDirLab[fplane];//根据邻居节点与Burgers向量的点积计算新的节点位置，并在选定的滑移方向上添加一个小的偏移，以确保节点在新的平面上有足够的移动，以便进行交滑移事件的评估
                        
                        // It looks like we should do the cross-slip, but to
                        // be sure, we need to move the nodes and evaluate
                        // the force on node in the new configuration. If
                        // it appears the node will not continue to move out
                        // on the new plane, skip the cross-slip event.
                        
                        // Compute force on segment by moving the node positions
                        // within the temporary, local SplitNet instance
                        // Override nodes i and n2 and set new positions
                        splitnet.nid[0] = i;
                        splitnet.nodes[0] = nodes[i];
                        splitnet.conn[0] = conn[i];
                        splitnet.nid[1] = n2;
                        splitnet.nodes[1] = nodes[n2];
                        splitnet.conn[1] = conn[n2];
                        splitnet.nconn = 2;
                        
                        splitnet.nodes[0].pos = nodep;
                        splitnet.nodes[1].pos = nbr2p;//在临时的SplitNet实例中覆盖当前节点和邻居节点2，并设置它们的新位置，以便评估在新的配置下节点上的力
                        
                        // Evaluate force on temporary configuration
                        Vec3 newforce = force->node_force(system, &splitnet, i, team);
                        double newfdotglide = dot(newforce, glideDirLab[fplane]);
                        
                        if ((SIGN(newfdotglide) * SIGN(fdotglide)) < 0.0) {
                            return;
                        }//如果在新的配置下节点上的力在选定的滑移方向上的分量与原来的力的分量符号相反，说明节点不会继续沿着新的平面移动，因此跳过这个交滑移事件
                        
                        // Save the new node positions and plane
                        if (tid == 0) {
                            events(lid).type = 0;
                            events(lid).p0 = nodep;
                            events(lid).p1 = nbr2p;
                            events(lid).plane = newplane;
                        }//如果满足交滑移事件的条件，并且在新的配置下节点上的力的分量符号没有发生改变，则保存新的节点位置和新的平面信息，以便后续执行交滑移事件的操作
                    }
                } else {
                    // Neighbor 1 can be moved, so proceed with the
                    // cross-slip operation.
                    
                    double vec1dotb = dot(vec1, burg);
                    nbr1p = nbr2p + vec1dotb * burg;
                    
                    double vec3dotb = dot(vec3, burg);
                    nodep = nbr2p + vec3dotb * burg;
                    
                    double fdotglide = dot(fLab, glideDirLab[fplane]);
                    double tmp = areamin / fabs(vec1dotb) * 2.0 * (1.0 + eps) * SIGN(fdotglide);
                    nodep += tmp * glideDirLab[fplane];//根据邻居节点1与Burgers向量的点积计算新的邻居节点1的位置，根据邻居节点2与Burgers向量的点积计算新的当前节点的位置，并在选定的滑移方向上添加一个小的偏移，以确保节点在新的平面上有足够的移动，以便进行交滑移事件的评估
                    
                    // It looks like we should do the cross-slip, but to
                    // be sure, we need to move the nodes and evaluate
                    // the force on node in the new configuration. If
                    // it appears the node will not continue to move out
                    // on the new plane, skip the cross-slip event.
                    
                    // Compute force on segment by moving the node positions
                    // within the temporary, local SplitNet instance
                    // Override nodes i and n1 and set new positions
                    splitnet.nid[0] = i;
                    splitnet.nodes[0] = nodes[i];
                    splitnet.conn[0] = conn[i];
                    splitnet.nid[1] = n1;
                    splitnet.nodes[1] = nodes[n1];
                    splitnet.conn[1] = conn[n1];
                    splitnet.nconn = 2;
                    
                    splitnet.nodes[0].pos = nodep;
                    splitnet.nodes[1].pos = nbr1p;
                    
                    Vec3 newforce = force->node_force(system, &splitnet, i, team);
                    double newfdotglide = dot(newforce, glideDirLab[fplane]);//在临时的SplitNet实例中覆盖当前节点和邻居节点1，并设置它们的新位置，以便评估在新的配置下节点上的力，并计算新的力在选定的滑移方向上的分量
                    
                    if ((SIGN(newfdotglide) * SIGN(fdotglide)) < 0.0) {
                        return;
                    }
                    
                    // Save the new node positions and plane
                    if (tid == 0) {
                        events(lid).type = 1;
                        events(lid).p0 = nodep;
                        events(lid).p1 = nbr1p;
                        events(lid).plane = newplane;
                    }//如果在新的配置下节点上的力在选定的滑移方向上的分量与原来的力的分量符号相反，说明节点不会继续沿着新的平面移动，因此跳过这个交滑移事件；如果满足交滑移事件的条件，并且在新的配置下节点上的力的分量符号没有发生改变，则保存新的节点位置和新的平面信息，以便后续执行交滑移事件的操作
                }
            
            } else if (seg1_is_screw && (plane1 != plane2) && (plane2 == fplane) &&
                       (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane1])+fnodeThreshold))) {
                
                // Zipper condition met for first segment.  If the first
                // neighbor is either not pinned or pinned but already
                // sufficiently aligned, proceed with the cross-slip event
                
                bool pinned1 = node_pinned(system, net, n1, plane1, glideDirLab, numGlideDir);//如果第一个线段接近于螺旋位错，并且两个线段所在的平面不同，并且第二个线段所在的平面等于选定的滑移方向所在的平面，并且力在选定的滑移方向上的分量大于一个阈值，则满足第一个线段的交滑移条件，此时需要判断邻居节点1是否被固定在原位，如果没有被固定或者被固定但已经足够对齐，则可以进行交滑移操作，否则跳过这个交滑移事件
                
                if ((!pinned1) || ((testmax2-test2) < (eps*eps*burgSize*burgSize))) {
                    
                    // Before 'zippering' a segment, try a quick sanity check
                    // to see if it makes sense.  If the force on the segment
                    // to be 'zippered' is less than 5% larger on the new
                    // plane than the old plane, leave the segment alone.
                    
                    // Compute force on segment by creating a temporary new node
                    // within the temporary, local SplitNet instance
                    Vec3 pmid = 0.5 * (nodep + nbr1p);
                    int nnew = splitnet.split_seg(s1, pmid);
                    
                    Vec3 newSegForce = force->node_force(system, &splitnet, nnew, team);
                    
                    double zipperThreshold = noiseFactor * shearModulus *
                                             burgSize *  L1;
                    double f1dotplane1 = fabs(dot(newSegForce, glideDirLab[plane1]));
                    double f1dotplanef = fabs(dot(newSegForce, newplane));//在临时的SplitNet实例中创建一个新的节点，位于当前节点和邻居节点1的中点位置，并计算在这个新的配置下线段上的力，然后比较这个力在原来平面上的分量和在新平面上的分量，如果在新平面上的分量没有比在原来平面上的分量大很多，则认为不值得进行交滑移操作，因此跳过这个事件
                    
                    if (f1dotplanef < zipperThreshold + f1dotplane1) {
                        return;
                    }
                    
                    if (!pinned1) {
                        double vec2dotb = dot(vec2, burg);
                        nbr1p = nodep - vec2dotb * burg;
                    }
                    
                    // Save the new node position and plane
                    if (tid == 0) {
                        events(lid).type = 2;
                        events(lid).p1 = nbr1p;
                        events(lid).plane = newplane;
                    }//如果满足第一个线段的交滑移条件，并且在新的配置下线段上的力在新平面上的分量比在原来平面上的分量大很多，则保存新的邻居节点1的位置和新的平面信息，以便后续执行交滑移事件的操作
                }
                
            } else if (seg2_is_screw && (plane1 != plane2) && (plane1 == fplane) &&
                       (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane2])+fnodeThreshold))) {
                
                // Zipper condition met for second segment
                
                bool pinned2 = node_pinned(system, net, n2, plane2, glideDirLab, numGlideDir);//如果第二个线段接近于螺旋位错，并且两个线段所在的平面不同，并且第一个线段所在的平面等于选定的滑移方向所在的平面，并且力在选定的滑移方向上的分量大于一个阈值，则满足第二个线段的交滑移条件，此时需要判断邻居节点2是否被固定在原位，如果没有被固定或者被固定但已经足够对齐，则可以进行交滑移操作，否则跳过这个交滑移事件
                
                if ((!pinned2) || ((testmax2-test2) < (eps*eps*burgSize*burgSize))) {
                    
                    // Compute force on segment by creating a temporary new node
                    // within the temporary, local SplitNet instance
                    Vec3 pmid = 0.5 * (nodep + nbr2p);
                    int nnew = splitnet.split_seg(s2, pmid);
                    
                    Vec3 newSegForce = force->node_force(system, &splitnet, nnew, team);
                    
                    double zipperThreshold = noiseFactor * shearModulus *
                                             burgSize *  L2;
                    double f1dotplane2 = fabs(dot(newSegForce, glideDirLab[plane2]));
                    double f1dotplanef = fabs(dot(newSegForce, newplane));

                    if (f1dotplanef < zipperThreshold + f1dotplane2) {
                        return;
                    }//在临时的SplitNet实例中创建一个新的节点，位于当前节点和邻居节点2的中点位置，并计算在这个新的配置下线段上的力，然后比较这个力在原来平面上的分量和在新平面上的分量，如果在新平面上的分量没有比在原来平面上的分量大很多，则认为不值得进行交滑移操作，因此跳过这个事件
                    
                    if (!pinned2) {
                        double vec3dotb = dot(vec3, burg);
                        nbr2p = nodep - vec3dotb * burg;
                    }//如果满足第二个线段的交滑移条件，并且在新的配置下线段上的力在新平面上的分量比在原来平面上的分量大很多，则保存新的邻居节点2的位置和新的平面信息，以便后续执行交滑移事件的操作
                    
                    // Save the new node position and plane
                    if (tid == 0) {
                        events(lid).type = 3;
                        events(lid).p1 = nbr2p;
                        events(lid).plane = newplane;
                    }//如果满足第二个线段的交滑移条件，并且在新的配置下线段上的力在新平面上的分量比在原来平面上的分量大很多，则保存新的邻居节点2的位置和新的平面信息，以便后续执行交滑移事件的操作
                }
            }
        }
    };
    
    void handle(System* system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_CROSSSLIP].start();
        
        if (system->crystal.type != FCC_CRYSTAL && system->crystal.type != BCC_CRYSTAL)
            ExaDiS_fatal("Error: CrossSlipParallel only implemented for FCC and BCC crystals\n");
        
        if (!system->crystal.use_glide_planes)//如果系统的晶体没有使用滑移平面选项，则无法进行交滑移操作，因此抛出一个错误提示
            ExaDiS_fatal("Error: CrossSlipParallel requires use_glide_planes option\n");
        
        int active_net = system->net_mngr->get_active();//获取当前活跃的网络索引
        DeviceDisNet* net = system->get_device_network();//获取当前活跃的网络实例
        
        // Initialize the FindCrossSlipEvents structure
        FindCrossSlipEvents* cs = exadis_new<FindCrossSlipEvents>(system, net, force);
        
        // Identify nodes attached to screw segments that need
        // to be considered for a cross-slip event
        Kokkos::resize(cs->count, 1);
        Kokkos::deep_copy(cs->csnodes, 0);
        Kokkos::resize(cs->csnodes, net->Nnodes_local);
        
        Kokkos::parallel_for(net->Nnodes_local, *cs);
        Kokkos::fence();
        
        int numcsnodes = cs->count(0);
        Kokkos::resize(cs->csnodes, numcsnodes);//根据之前的并行循环计算得到需要考虑交滑移事件的节点数量，并将csnodes视图的大小调整为这个数量，以便后续处理交滑移事件时只关注这些节点
        
        // If we need a neighbor list, let's build a contiguous
        // one for only the subset of split nodes so that access 
        // on device will be much faster
        double cutoff = system->neighbor_cutoff;
        NeighborList* neilist;
        if (cutoff > 0.0) {
            NeighborBox* neighbox = exadis_new<NeighborBox>(system, cutoff, Neighbor::NeiSeg);
            // Build a neighbor list of the nodes wrt to the segs
            neilist = neighbox->build_neighbor_list(system, net, Neighbor::NeiNode, cs->csnodes);
            cs->neilist = neilist;
            exadis_delete(neighbox);
        }//如果系统的邻居截断距离大于0，则需要构建一个邻居列表来加速后续处理交滑移事件时对节点和线段的访问，首先创建一个NeighborBox实例，并使用它来构建一个针对csnodes中节点的邻居列表，最后将这个邻居列表存储在FindCrossSlipEvents结构中，以便后续使用
        
        // Find all cross-slip events that we need to handle.
        // This is done in parallel where each node previously
        // identified is now assigned a team of threads.
        Kokkos::resize(cs->events, numcsnodes);
        Kokkos::parallel_for(Kokkos::TeamPolicy<>(numcsnodes, Kokkos::AUTO), *cs);
        Kokkos::fence();//使用Kokkos的并行循环来处理每个需要考虑交滑移事件的节点，每个节点被分配一个线程团队来执行FindCrossSlipEvents结构中的operator()函数，以便判断是否满足交滑移事件的条件，并将结果存储在events视图中
        
        // We are done with determining the cross-slip events, now
        // execute the changes. We do this in serial for simplicity.
        auto h_events = Kokkos::create_mirror_view(cs->events);
        auto h_csnodes = Kokkos::create_mirror_view(cs->csnodes);//创建events和csnodes视图的镜像视图，以便在主机上访问它们的内容
        
        // We did not make any changes to the network yet, so
        // let's avoid making unnecessary memory copies
        system->net_mngr->set_active(active_net);
        SerialDisNet* network = system->get_serial_network();
        
        std::vector<int> eventflag(numcsnodes, 0);//创建一个整数向量eventflag，用于标记每个交滑移事件的状态，初始值为0，表示事件尚未执行；当事件被执行后，将对应的标记设置为1；如果事件被跳过，则将对应的标记设置为-1，以便后续处理时能够区分不同状态的事件
        // -1: skip, 0: not executed, 1: done
        for (int k = 0; k < numcsnodes; k++) {
            if (h_events(k).type < 0) eventflag[k] = -1;
        }
        
        // Start with the zipper events
        for (int k = 0; k < numcsnodes; k++) {
            if (eventflag[k] != 0) continue;
            
            CrossSlipEvent& event = h_events(k);
            int type = event.type;
            if (type < 2) continue; // zipper events = {2,3} 这个是根据之前保存的事件类型来区分交滑移事件和拉链事件，首先处理拉链事件，即事件类型为2或3的情况
            
            int i = h_csnodes(k); // node id
            int n = network->conn[i].node[type-2]; // neighbor id
            int s = network->conn[i].seg[type-2]; // seg id
            Vec3 newplane = event.plane;
            
            // If the current zipper event affects a node that was
            // involved in a previous zipper event, we will only allow
            // the current event to proceeed if the glide planes
            // for the two events match.
            bool skip = 0;
            for (int l = 0; l < k; l++) {
                if (eventflag[l] != 1) continue;
                
                CrossSlipEvent& prev_event = h_events(l);
                int prev_type = prev_event.type;
                if (prev_type < 2) continue; // zipper events = {2,3}
                
                int prev_i = h_csnodes(l); // node id
                int prev_n = network->conn[prev_i].node[prev_type-2]; // neighbor id
                
                if ((i == prev_n) || (n == prev_i) || (n == prev_n)) {
                    if (cross(prev_event.plane, newplane).norm2() > 1.0e-3) {
                        skip = 1;
                        break;
                    }
                }
            }//对于当前的拉链事件，如果它涉及的节点之前已经参与过一个拉链事件，那么我们将只允许当前事件继续进行，如果两个事件的滑移平面匹配，否则跳过当前事件；具体来说，我们遍历之前的事件，如果之前的事件是一个拉链事件，并且它涉及的节点与当前事件涉及的节点相同或者相邻，那么我们比较两个事件的滑移平面，如果它们之间的夹角大于一个小的阈值，则认为它们不匹配，因此跳过当前事件
            if (skip) continue;
            eventflag[k] = 1;
            
            // Reposition neighbor node
            network->move_node(n, event.p1, system->dEp);
            // Update the segment glide plane
            update_seg_plane(network, s, newplane);
        }//对于满足条件的拉链事件，我们首先将对应的标记设置为1，表示事件已经执行，然后重新定位邻居节点的位置，并更新线段的滑移平面，以完成拉链事件的操作
        
        // Continue with the cross-slip events
        for (int k = 0; k < numcsnodes; k++) {
            if (eventflag[k] != 0) continue;//对于剩下的交滑移事件，即事件类型为0或1的情况，我们首先检查它们是否满足条件，如果满足条件则执行交滑移事件的操作，否则跳过；具体来说，我们遍历之前的事件，如果之前的事件是一个交滑移事件，并且它涉及的节点与当前事件涉及的节点相同或者相邻，那么我们比较两个事件的滑移平面，如果它们之间的夹角大于一个小的阈值，则认为它们不匹配，因此跳过当前事件；如果满足条件，我们首先将对应的标记设置为1，表示事件已经执行，然后重新定位当前节点和邻居节点的位置，并更新两个线段的滑移平面，以完成交滑移事件的操作
            
            CrossSlipEvent& event = h_events(k);
            int type = event.type;
            if (type > 1) continue; // cross-slip events = {0,1}
            
            int i = h_csnodes(k); // node id
            int n = network->conn[i].node[1-type]; // neighbor id
            int n1 = network->conn[i].node[0];
            int n2 = network->conn[i].node[1];
            int s1 = network->conn[i].seg[0]; // seg 1 id
            int s2 = network->conn[i].seg[1]; // seg 2 id
            Vec3 newplane = event.plane;//获取当前交滑移事件涉及的节点和线段的信息，包括节点ID、邻居节点ID、连接的线段ID以及新的平面信息
            
            // If the current cross-slip event affects a node that was
            // involved in a previous cross-slip event, we will only allow
            // the current event to proceeed if the glide planes
            // for the two events match.
            bool skip = 0;
            for (int l = 0; l < k; l++) {
                if (eventflag[l] != 1) continue;
                
                CrossSlipEvent& prev_event = h_events(l);
                int prev_type = prev_event.type;
                int prev_i = h_csnodes(l); // node id
                
                if (prev_type > 1) {
                    // check against zipper events = {2,3}
                    int prev_n = network->conn[prev_i].node[prev_type-2]; // neighbor id
                    
                    if ((i == prev_n) || (n1 == prev_i) || (n1 == prev_n) || 
                        (n2 == prev_i) || (n2 == prev_n)) {
                        if (cross(prev_event.plane, newplane).norm2() > 1.0e-3) {
                            skip = 1;
                            break;
                        }
                    }//如果之前的事件是一个拉链事件，并且它涉及的节点与当前事件涉及的节点相同或者相邻，那么我们比较两个事件的滑移平面，如果它们之间的夹角大于一个小的阈值，则认为它们不匹配，因此跳过当前事件
                } else {
                    // check against cross-slip events = {0,1}
                    int prev_n1 = network->conn[prev_i].node[0];
                    int prev_n2 = network->conn[prev_i].node[1];
                    
                    if ((i == prev_n1) || (i == prev_n2) ||
                        (n1 == prev_i) || (n1 == prev_n1) || (n1 == prev_n2) || 
                        (n2 == prev_i) || (n2 == prev_n1) || (n2 == prev_n2)) {
                        if (cross(prev_event.plane, newplane).norm2() > 1.0e-3) {
                            skip = 1;
                            break;
                        }
                    }//如果之前的事件是一个交滑移事件，并且它涉及的节点与当前事件涉及的节点相同或者相邻，那么我们比较两个事件的滑移平面，如果它们之间的夹角大于一个小的阈值，则认为它们不匹配，因此跳过当前事件
                }
            }
            if (skip) continue;
            eventflag[k] = 1;
            
            // Reposition nodes
            network->move_node(i, event.p0, system->dEp);
            network->move_node(n, event.p1, system->dEp);
            // Update segments glide plane
            update_seg_plane(network, s1, newplane);
            update_seg_plane(network, s2, newplane);
        }//对于满足条件的交滑移事件，我们首先将对应的标记设置为1，表示事件已经执行，然后重新定位当前节点和邻居节点的位置，并更新两个线段的滑移平面，以完成交滑移事件的操作
        
        
        if (cutoff > 0.0)
            exadis_delete(neilist);
            
        exadis_delete(cs);
        
        Kokkos::fence();
        system->timer[system->TIMER_CROSSSLIP].stop();
    }
    
    const char* name() { return "CrossSlipParallel"; }
};

} // namespace ExaDiS

#endif

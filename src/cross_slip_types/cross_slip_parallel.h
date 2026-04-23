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
template<class F>//F代表力计算内核的类型
class CrossSlipParallel : public CrossSlip {
private:
    F* force;//指向力计算内核的指针
    
public:
    CrossSlipParallel(System* system, Force* _force)//构造函数，接受系统指针和力计算内核指针作为参数
    {
        // Check and assign force kernel
        force = dynamic_cast<F*>(_force);//尝试将_force转换为F类型的指针，并赋值给force成员变量
        if (force == nullptr)//如果转换失败，force将为nullptr，说明传入的_force类型与F不兼容
            ExaDiS_fatal("Error: inconsistent force type in CrossSlipParallel\n");//输出错误信息并终止程序
    }
    
    /*-----------------------------------------------------------------------
     *    Struct:     CrossSlipEvent
     *                Structure to hold cross-slip event information.
     *---------------------------------------------------------------------*/
    struct CrossSlipEvent {
        int type; 
        Vec3 p0, p1;
        Vec3 plane;
    };//定义了一个结构体CrossSlipEvent，用于存储交叉滑移事件的信息，包括事件类型、两个位置向量p0和p1，以及一个平面向量plane。
    
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
        NeighborList* neilist;//指向邻居列表的指针
        
        double eps, thetacrit, sthetacrit, s2thetacrit;//定义了一些用于交叉滑移事件计算的参数，包括一个小的数值eps、一个临界角度thetacrit，以及其正弦值sthetacrit和s2thetacrit。
        double shearModulus, areamin;//定义了剪切模量shearModulus和最小面积areamin，用于交叉滑移事件的计算。
        double noiseFactor, weightFactor;//定义了噪声因子noiseFactor和权重因子weightFactor，用于调整交叉滑移事件的计算。
        
        Mat33 R, Rinv;//定义了两个3x3矩阵R和Rinv，分别表示晶体的旋转矩阵和其逆矩阵，用于将向量在实验室坐标系和晶体坐标系之间转换。
        
        Kokkos::View<int*, T_memory_shared> count;
        Kokkos::View<int*, T_memory_space> csnodes;
        Kokkos::View<CrossSlipEvent*, T_memory_shared> events;
        
        FindCrossSlipEvents(System* _system, DeviceDisNet* _net, F* _force) :
        system(_system), net(_net), force(_force)
        {
            eps = 1e-6;
            if (system->crystal.type == FCC_CRYSTAL)
                thetacrit = 2.0 / 180.0 * M_PI;//对于FCC晶体，临界角度thetacrit被设置为2度（转换为弧度），用于判断哪些段落接近螺旋形态。
            else if (system->crystal.type == BCC_CRYSTAL)
                thetacrit = 0.5 / 180.0 * M_PI;//对于BCC晶体，临界角度thetacrit被设置为0.5度（转换为弧度），用于判断哪些段落接近螺旋形态。
            sthetacrit = sin(thetacrit);//计算thetacrit的正弦值sthetacrit，用于后续的计算中。
            s2thetacrit = sthetacrit * sthetacrit;//计算sthetacrit的平方s2thetacrit，用于后续的计算中。
            shearModulus = system->params.MU;//从系统参数中获取剪切模量shearModulus，用于交叉滑移事件的计算。
            
            areamin = 2.0 * system->params.rtol * system->params.maxseg;//计算最小面积areamin，基于系统参数中的rtol和maxseg，用于交叉滑移事件的计算。
            areamin = MIN(areamin, system->params.minseg * system->params.minseg * sqrt(3.0) / 4.0);//将areamin限制在一个最大值，基于系统参数中的minseg，确保交叉滑移事件的计算不会产生过小的面积。
            
            noiseFactor = 1e-5;//设置噪声因子noiseFactor为1e-5，用于调整交叉滑移事件的计算，可能用于引入一些随机性以避免数值问题。
            weightFactor = 1.0;//设置权重因子weightFactor为1.0，用于调整交叉滑移事件的计算，可能用于平衡不同因素在计算中的影响。
            
            R = system->crystal.R;
            Rinv = system->crystal.Rinv;
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const int& i) const
        {
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            auto conn = net->get_conn();//获取网络中的节点、段落和连接信息，以便在后续的计算中使用这些信息来判断哪些节点可能发生交叉滑移事件。
            auto cell = net->cell;
            
            if (conn[i].num != 2) return;//如果连接数不等于2，说明该节点不是一个长螺旋段的中间节点，因此直接返回，不进行交叉滑移事件的判断。
            if (nodes[i].constraint != UNCONSTRAINED) return;//如果节点受到约束，说明它不能自由移动，因此直接返回，不进行交叉滑移事件的判断。
            
            int s = conn[i].seg[0];//获取连接的第一个段落索引s，并从段落信息中获取Burgers向量burg和其大小burgSize。然后将Burgers向量归一化，并转换到晶体坐标系中，得到burgCrystal。接下来，根据晶体类型（FCC或BCC）进行不同的判断，确定该段落是否满足交叉滑移事件的条件。
            Vec3 burg = segs[s].burg;
            double burgSize = burg.norm();
            burg = burg.normalized();
            
            Vec3 burgCrystal = Rinv * burg;
            burgCrystal = burgCrystal.normalized();//将Burgers向量转换到晶体坐标系中，并归一化，以便后续的判断中使用晶体坐标系下的Burgers向量进行比较。
            
            if (system->crystal.type == FCC_CRYSTAL) {
            
                // Only consider glide dislocations. If the Burgers vector is not
                // a [1 1 0] type, ignore it.
                if (!((fabs(fabs(burgCrystal.x)-fabs(burgCrystal.y)) < eps) &&
                      (fabs(burgCrystal.z) < eps)) &&
                    !((fabs(fabs(burgCrystal.y)-fabs(burgCrystal.z)) < eps) &&
                      (fabs(burgCrystal.x) < eps )) &&
                    !((fabs(fabs(burgCrystal.z)-fabs(burgCrystal.x)) < eps) &&
                      (fabs(burgCrystal.y) < eps))) {
                    return;
                }//对于FCC晶体，只有当Burgers向量在晶体坐标系下满足特定的条件时，才考虑发生交叉滑移事件。具体来说，Burgers向量必须接近于[1 1 0]类型，即其中两个分量的绝对值相等且第三个分量接近于零。如果不满足这个条件，则直接返回，不进行交叉滑移事件的判断。

                if ((fabs(burgCrystal.x) < eps) && (fabs(burgCrystal.y) < eps) &&
                    (fabs(burgCrystal.z) < eps)) {
                    return;
                }//如果Burgers向量在晶体坐标系下的所有分量都接近于零，说明它不是一个有效的Burgers向量，因此直接返回，不进行交叉滑移事件的判断。
                
                // Also test that the segment resides on a (1 1 1) plane, since these
                // are the only planes where cross-slip occurs.
                Vec3 plane = segs[s].plane.normalized();
                Vec3 planeCrystal = Rinv * plane;
                if ((fabs(fabs(planeCrystal.x) - fabs(planeCrystal.y)) > eps) ||
                    (fabs(fabs(planeCrystal.y) - fabs(planeCrystal.z)) > eps)) {
                    return; // not a {111} plane
                }//此外，还需要判断该段落是否位于(1 1 1)类型的晶面上，因为交叉滑移事件只发生在这些晶面上。具体来说，段落的平面向量在晶体坐标系下必须满足特定的条件，即其中两个分量的绝对值相等。如果不满足这个条件，则直接返回，不进行交叉滑移事件的判断。
                
            } else if (system->crystal.type == BCC_CRYSTAL) {
                
                // Only consider <111> dislocations
                if (fabs(burgCrystal.x * burgCrystal.y * burgCrystal.z) < eps) {
                    return;
                }
                
            }//对于BCC晶体，只有当Burgers向量在晶体坐标系下满足特定的条件时，才考虑发生交叉滑移事件。具体来说，Burgers向量的三个分量的乘积必须大于一个小的数值eps，即它们都不能接近于零。如果不满足这个条件，则直接返回，不进行交叉滑移事件的判断。
            
            int n1 = conn[i].node[0];
            int n2 = conn[i].node[1];//获取连接的两个节点索引n1和n2，并计算当前节点与这两个邻居节点之间的向量关系，以判断该节点是否接近于螺旋形态，从而确定是否满足交叉滑移事件的条件。
            
            Vec3 nodep = nodes[i].pos;
            Vec3 nbr1p = cell.pbc_position(nodep, nodes[n1].pos);
            Vec3 nbr2p = cell.pbc_position(nodep, nodes[n2].pos);//获取当前节点的位置nodep，以及两个邻居节点的位置nbr1p和nbr2p，并考虑周期性边界条件（PBC）来计算它们之间的相对位置，以便后续的判断中使用这些位置关系来确定该节点是否接近于螺旋形态。
            
            // If the node is a point on a long screw then we can consider
            // it for possible cross slip.
            Vec3 vec1 = nbr1p - nbr2p;
            Vec3 vec2 = nodep - nbr1p;
            Vec3 vec3 = nodep - nbr2p;//计算当前节点与两个邻居节点之间的向量关系，vec1表示两个邻居节点之间的向量，vec2表示当前节点与第一个邻居节点之间的向量，vec3表示当前节点与第二个邻居节点之间的向量。这些向量关系将用于后续的判断中，以确定该节点是否接近于螺旋形态，从而判断是否满足交叉滑移事件的条件。
            
            // Calculate some test conditions
            double test1 = dot(vec1, burg);
            double test2 = dot(vec2, burg);
            double test3 = dot(vec3, burg);//计算当前节点与两个邻居节点之间的向量关系在Burgers向量方向上的投影，得到test1、test2和test3。这些投影值将用于后续的判断中，以确定该节点是否接近于螺旋形态，从而判断是否满足交叉滑移事件的条件。
            
            test1 = test1 * test1;
            test2 = test2 * test2;
            test3 = test3 * test3;

            double testmax1 = dot(vec1, vec1);
            double testmax2 = dot(vec2, vec2);
            double testmax3 = dot(vec3, vec3);//计算当前节点与两个邻居节点之间的向量关系的平方长度，得到testmax1、testmax2和testmax3。这些长度值将用于后续的判断中，以确定该节点是否接近于螺旋形态，从而判断是否满足交叉滑移事件的条件。
            
            // Set up the tests to see if this dislocation is close enough to
            // screw to be considered for cross slip.  For a segment to be close
            //to screw it must be within 2*thetacrit defined above
            bool seg1_is_screw = ((testmax2 - test2) < (testmax2 * s2thetacrit));//根据之前定义的临界角度thetacrit和其正弦值s2thetacrit，判断当前节点与第一个邻居节点之间的向量关系是否接近于螺旋形态。具体来说，如果testmax2 - test2小于testmax2 * s2thetacrit，说明当前节点与第一个邻居节点之间的向量关系接近于螺旋形态，即该段落可能是一个长螺旋段的一部分，因此将seg1_is_screw设置为true。
            bool seg2_is_screw = ((testmax3 - test3) < (testmax3 * s2thetacrit));
            bool bothseg_are_screw =
                (((testmax2 - test2) < (4.0 * testmax2 * s2thetacrit)) &&
                 ((testmax3 - test3) < (4.0 * testmax3 * s2thetacrit)) &&
                 ((testmax1 - test1) < (testmax1 * s2thetacrit)));
                 
            if (seg1_is_screw || seg2_is_screw || bothseg_are_screw) {
                int idx = Kokkos::atomic_fetch_add(&count(0), 1);
                csnodes(idx) = i;
            }//如果当前节点与第一个邻居节点之间的向量关系接近于螺旋形态（seg1_is_screw为true），或者当前节点与第二个邻居节点之间的向量关系接近于螺旋形态（seg2_is_screw为true），或者当前节点与两个邻居节点之间的向量关系都接近于螺旋形态（bothseg_are_screw为true），则说明该节点可能发生交叉滑移事件。此时，使用原子操作将count(0)加1，并将当前节点的索引i存储在csnodes(idx)中，以便后续的并行计算中使用这些索引来处理可能发生交叉滑移事件的节点。
        }
        
        KOKKOS_INLINE_FUNCTION
        void operator() (const team_handle& team) const
        {
            int tid = team.team_rank();
            int lid = team.league_rank();
            int i = csnodes(lid); // node id
            
            // Flag no event type
            if (tid == 0) events(lid).type = -1;
            
            auto nodes = net->get_nodes();
            auto segs = net->get_segs();
            auto conn = net->get_conn();//获取网络中的节点、段落和连接信息，以便在后续的计算中使用这些信息来处理可能发生交叉滑移事件的节点。
            auto cell = net->cell;
            
            // Recompute some info about the local node
            int s = conn[i].seg[0];
            Vec3 burg = segs[s].burg;
            double burgSize = burg.norm();
            burg = burg.normalized();//重新计算当前节点的相关信息，包括获取连接的第一个段落索引s，并从段落信息中获取Burgers向量burg和其大小burgSize。然后将Burgers向量归一化，以便后续的计算中使用归一化的Burgers向量进行判断和处理交叉滑移事件。
            
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
                 ((testmax1 - test1) < (testmax1 * s2thetacrit)));//重新计算当前节点与两个邻居节点之间的向量关系，并判断它们是否接近于螺旋形态，以便在后续的处理交叉滑移事件时使用这些信息来确定具体的交叉滑移事件类型和处理方式。
            
            // Since we will likely need to locally modify the network
            // let's first create a new configuration within a temporary, 
            // local SplitNet instance (from Topology)
            SplitDisNet splitnet(net, neilist);//由于我们可能需要局部修改网络结构，因此首先在一个临时的、局部的SplitNet实例中创建一个新的配置，以便在后续的处理交叉滑移事件时使用这个SplitNet实例来进行网络结构的修改和更新。
            
            // Compute the nodal force (initially in laboratory frame)
            Vec3 fLab = force->node_force(system, &splitnet, i, team);//计算当前节点的力，初始情况下是在实验室坐标系下计算的，以便在后续的处理交叉滑移事件时使用这个力来判断是否满足交叉滑移事件的条件，以及确定具体的交叉滑移事件类型和处理方式。
            
            // Set the force threshold for noise level within the code
            double L1 = sqrt(testmax2);
            double L2 = sqrt(testmax3);
            double fnodeThreshold = noiseFactor * shearModulus * burgSize * 
                                    0.5 * (L1 + L2);//设置代码中的噪声水平的力阈值fnodeThreshold，基于之前计算的当前节点与两个邻居节点之间的向量关系的长度L1和L2，以及系统参数中的剪切模量shearModulus、Burgers向量的大小burgSize和之前定义的噪声因子noiseFactor。这个力阈值将用于后续的判断中，以确定当前节点是否满足交叉滑移事件的条件，以及确定具体的交叉滑移事件类型和处理方式。
            
            Mat33 glideDirCrystal = Mat33().zero();
            int numGlideDir = 0; // Number of cross-slip glide directions 
            //数字为0的numGlideDir表示交叉滑移的滑移方向数量，具体的滑移方向将在后续的代码中根据晶体类型（FCC或BCC）进行判断和设置，以便在处理交叉滑移事件时使用这些滑移方向来确定新的滑移平面和滑移方向。
            if (system->crystal.type == FCC_CRYSTAL) {
                // Find which glide planes the segments are on
                // e.g. for burg = [ 1  1  0 ], the two glide directions are
                //                 [ 1 -1  2 ] and
                //                 [ 1 -1 -2 ]
                // Use Burgers vectors in crystal frame to generate initial glide
                // planes in crystal frame.
                numGlideDir = 2;//对于FCC晶体，交叉滑移的滑移方向数量numGlideDir被设置为2，因为对于一个特定的Burgers向量，例如[1 1 0]，存在两个可能的滑移方向，即[1 -1 2]和[1 -1 -2]。这些滑移方向将用于后续的处理交叉滑移事件时，来确定新的滑移平面和滑移方向。
                double tmp = 1.0;//使用晶体坐标系下的Burgers向量来生成初始的滑移平面。具体来说，对于每个分量，如果该分量的绝对值大于一个小的数值eps，则根据该分量的符号来设置对应的滑移方向分量为1.0或-1.0；如果该分量的绝对值小于eps，则将对应的滑移方向分量设置为2.0或-2.0。这样就得到了两个初始的滑移方向，存储在glideDirCrystal矩阵中。
                for (int j = 0; j < 3; j++) {
                    if (fabs(burgCrystal[j]) > eps) {
                        glideDirCrystal[0][j] = (burgCrystal[j]*tmp > 0) ? 1.0 : -1.0;
                        glideDirCrystal[1][j] = (burgCrystal[j]*tmp > 0) ? 1.0 : -1.0;
                        tmp = -1.0;
                    } else {
                        glideDirCrystal[0][j] =  2.0;
                        glideDirCrystal[1][j] = -2.0;
                    }
                }//这是根据晶体坐标系下的Burgers向量来生成初始的滑移平面。具体来说，对于每个分量，如果该分量的绝对值大于一个小的数值eps，则根据该分量的符号来设置对应的滑移方向分量为1.0或-1.0；如果该分量的绝对值小于eps，则将对应的滑移方向分量设置为2.0或-2.0。这样就得到了两个初始的滑移方向，存储在glideDirCrystal矩阵中。
                
                // Normalization
                glideDirCrystal[0] = sqrt(1.0/6.0) * glideDirCrystal[0];//对生成的初始滑移方向进行归一化处理。具体来说，将每个滑移方向向量乘以sqrt(1.0/6.0)，以确保它们的长度为1.0。这是因为对于FCC晶体，交叉滑移的滑移方向是特定的，并且需要满足一定的几何关系，因此需要进行归一化处理来确保它们具有正确的长度和方向。
                glideDirCrystal[1] = sqrt(1.0/6.0) * glideDirCrystal[1];
                
            } else if (system->crystal.type == BCC_CRYSTAL) {
                // Find which glide planes the segments are on. Initial
                // glidedir array contains glide directions in crystal frame
                // For BCC geometry burgCrystal should be of <1 1 1> type
                numGlideDir = 3;//对于BCC晶体，交叉滑移的滑移方向数量numGlideDir被设置为3，因为对于一个特定的Burgers向量，例如<1 1 1>类型，存在三个可能的滑移方向。这些滑移方向将用于后续的处理交叉滑移事件时，来确定新的滑移平面和滑移方向。
                Mat33 tmp33 = outer(burgCrystal, burgCrystal);
                for (int m = 0; m < 3; m++)
                    for (int n = 0; n < 3; n++)
                        glideDirCrystal[m][n] = ((m==n)-tmp33[m][n]) * sqrt(1.5);//这是根据晶体坐标系下的Burgers向量来生成初始的滑移平面。具体来说，对于BCC晶体，Burgers向量应该是<1 1 1>类型的，因此可以通过计算Burgers向量的外积来得到一个矩阵tmp33，然后根据这个矩阵来设置滑移方向矩阵glideDirCrystal。具体来说，对于每个分量，如果m等于n，则对应的滑移方向分量为(1 - tmp33[m][n]) * sqrt(1.5)；如果m不等于n，则对应的滑移方向分量为(- tmp33[m][n]) * sqrt(1.5)。这样就得到了三个初始的滑移方向，存储在glideDirCrystal矩阵中。

                // glideDirCrystal should now contain the three <112> type
                // directions that a screw dislocation may move in if glide
                // is restricted to <110> type glide planes
            }
            
            int s1 = conn[i].seg[0];//获取连接的第一个段落索引s1和第二个段落索引s2，并从段落信息中获取它们的平面向量segplane1和segplane2。然后将这些平面向量转换到晶体坐标系中，以便在后续的处理交叉滑移事件时使用这些平面向量来判断新的滑移平面和滑移方向。
            int s2 = conn[i].seg[1];
            Vec3 segplane1 = segs[s1].plane;
            Vec3 segplane2 = segs[s2].plane;//获取连接的第一个段落索引s1和第二个段落索引s2，并从段落信息中获取它们的平面向量segplane1和segplane2。这些平面向量将用于后续的处理交叉滑移事件时，来判断新的滑移平面和滑移方向。
            
            // Rotations
            Mat33 glideDirLab;
            for (int j = 0; j < 3; j++)
                glideDirLab[j] = R * glideDirCrystal[j];//将晶体坐标系下的滑移方向转换到实验室坐标系中，得到glideDirLab矩阵，以便在后续的处理交叉滑移事件时使用这些滑移方向来确定新的滑移平面和滑移方向。
            segplane1 = Rinv * segplane1;
            segplane2 = Rinv * segplane2;
            Vec3 fCrystal = Rinv * fLab;
            
            Vec3 tmp3  = glideDirCrystal * segplane1;//计算滑移方向与段落平面之间的关系，得到tmp3、tmp3B和tmp3C。这些值将用于后续的处理交叉滑移事件时，来判断新的滑移平面和滑移方向，以及确定具体的交叉滑移事件类型和处理方式。
            Vec3 tmp3B = glideDirCrystal * segplane2;
            Vec3 tmp3C = glideDirCrystal * fCrystal;
            
            // For FCC there are only two slip planes for screw dislocation
            int plane1 = 0;
            int plane2 = 0;
            int fplane = 0;
            
            for (int j = 1; j < numGlideDir; j++) {
                plane1 = (fabs(tmp3[j])  < fabs(tmp3[plane1]) ) ? j : plane1;
                plane2 = (fabs(tmp3B[j]) < fabs(tmp3B[plane2])) ? j : plane2;
                fplane = (fabs(tmp3C[j]) > fabs(tmp3C[fplane])) ? j : fplane;
            }//对于FCC晶体，由于只有两个滑移平面，因此我们需要确定当前节点所在的段落平面与哪个滑移方向更接近，以及当前节点的力在滑移方向上的分量与哪个滑移方向更接近。具体来说，我们通过比较tmp3、tmp3B和tmp3C的绝对值来确定plane1、plane2和fplane，分别表示当前节点所在的段落平面与哪个滑移方向更接近，以及当前节点的力在滑移方向上的分量与哪个滑移方向更接近。
            
            // Calculate the new plane in the lab frame
            Vec3 newplane = cross(burg, glideDirLab[fplane]).normalized();//根据Burgers向量和当前节点的力在滑移方向上的分量所对应的滑移方向，计算新的滑移平面newplane，并将其归一化，以便在后续的处理交叉滑移事件时使用这个新的滑移平面来确定新的滑移方向和处理方式。
            
            if (bothseg_are_screw && (plane1 == plane2) && (plane1 != fplane) &&
                (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane1])+fnodeThreshold))) {
                
                // Both segments are close to screw and the average direction
                // is close to screw.
                
                // Determine if the neighbor nodes should be considered immobile
                bool pinned1 = node_pinned(system, net, n1, plane1, glideDirLab, numGlideDir);
                bool pinned2 = node_pinned(system, net, n2, plane2, glideDirLab, numGlideDir);
                
                if (pinned1) {
                    if ((!pinned2) || ((testmax1-test1) < (eps*eps*burgSize*burgSize))) {
                        
                        double vec1dotb = dot(vec1, burg);
                        double vec2dotb = dot(vec2, burg);//如果邻居节点n1被认为是固定的（pinned1为true），则需要进一步判断邻居节点n2是否可以移动。如果邻居节点n2没有被固定（pinned2为false），或者当前节点与两个邻居节点之间的向量关系非常接近于螺旋形态（testmax1 - test1小于一个小的数值eps乘以Burgers向量大小的平方），则说明邻居节点n2可以被移动，因此继续进行交叉滑移操作。
                        
                        if (!pinned2) {
                            // Neighbor 2 can be moved, so proceed with the
                            // cross-slip operation.
                            nbr2p = nbr1p - vec1dotb * burg;
                        }//如果邻居节点n2没有被固定（pinned2为false），则说明它可以被移动，因此继续进行交叉滑移操作。具体来说，根据当前节点与第一个邻居节点之间的向量关系在Burgers向量方向上的投影vec1dotb，计算新的邻居节点n2的位置nbr2p，使其与当前节点在新的滑移平面上对齐。
                        
                        // If neighbor2 is pinned, it is already perfectly
                        // aligned with neighbor1 in the screw direction
                        // so there is no need to move it.
                        nodep = nbr1p + vec2dotb * burg;
                        
                        double fdotglide = dot(fLab, glideDirLab[fplane]);
                        double tmp = areamin / fabs(vec1dotb) * 2.0 * (1.0 + eps) * SIGN(fdotglide);
                        nodep += tmp * glideDirLab[fplane];//如果邻居节点n2被认为是固定的（pinned2为true），则说明它已经在螺旋方向上与邻居节点n1完美对齐，因此不需要移动它。此时，根据当前节点与第一个邻居节点之间的向量关系在Burgers向量方向上的投影vec2dotb，计算新的当前节点的位置nodep，使其与邻居节点n1在新的滑移平面上对齐。然后根据当前节点的力在滑移方向上的分量fdotglide，计算一个临时值tmp，并将其乘以滑移方向来调整当前节点的位置，以确保它在新的滑移平面上具有正确的位移。
                        
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
                        splitnet.nconn = 2;//在临时的、局部的SplitNet实例中，通过移动节点位置来计算新的配置下的力，以判断是否满足交叉滑移事件的条件。具体来说，覆盖当前节点i和邻居节点n2，并设置它们的新位置nodep和nbr2p，然后计算当前节点在新的配置下的力newforce，并将其在滑移方向上的分量newfdotglide与之前计算的fdotglide进行比较，以判断当前节点是否会继续在新的滑移平面上移动。如果它们的符号相反，说明当前节点不会继续在新的滑移平面上移动，因此跳过交叉滑移事件。
                        
                        splitnet.nodes[0].pos = nodep;
                        splitnet.nodes[1].pos = nbr2p;//在临时的、局部的SplitNet实例中，覆盖当前节点i和邻居节点n2，并设置它们的新位置nodep和nbr2p，以便计算当前节点在新的配置下的力，并判断是否满足交叉滑移事件的条件。
                        
                        // Evaluate force on temporary configuration
                        Vec3 newforce = force->node_force(system, &splitnet, i, team);
                        double newfdotglide = dot(newforce, glideDirLab[fplane]);
                        
                        if ((SIGN(newfdotglide) * SIGN(fdotglide)) < 0.0) {
                            return;
                        }//如果当前节点在新的配置下的力在滑移方向上的分量newfdotglide与之前计算的fdotglide的符号相反，说明当前节点不会继续在新的滑移平面上移动，因此直接返回，不进行交叉滑移事件的处理。
                        
                        // Save the new node positions and plane
                        if (tid == 0) {
                            events(lid).type = 0;
                            events(lid).p0 = nodep;
                            events(lid).p1 = nbr2p;
                            events(lid).plane = newplane;
                        }//如果当前节点在新的配置下的力在滑移方向上的分量newfdotglide与之前计算的fdotglide的符号相同，说明当前节点会继续在新的滑移平面上移动，因此保存新的节点位置和新的滑移平面，以便后续的处理交叉滑移事件时使用这些信息来确定新的滑移方向和处理方式。
                    }
                } else {
                    // Neighbor 1 can be moved, so proceed with the
                    // cross-slip operation.
                    
                    double vec1dotb = dot(vec1, burg);
                    nbr1p = nbr2p + vec1dotb * burg;//如果邻居节点n1没有被固定（pinned1为false），则说明它可以被移动，因此继续进行交叉滑移操作。具体来说，根据当前节点与第一个邻居节点之间的向量关系在Burgers向量方向上的投影vec1dotb，计算新的邻居节点n1的位置nbr1p，使其与当前节点在新的滑移平面上对齐。
                    
                    double vec3dotb = dot(vec3, burg);
                    nodep = nbr2p + vec3dotb * burg;
                    
                    double fdotglide = dot(fLab, glideDirLab[fplane]);
                    double tmp = areamin / fabs(vec1dotb) * 2.0 * (1.0 + eps) * SIGN(fdotglide);//如果邻居节点n1没有被固定（pinned1为false），则说明它可以被移动，因此继续进行交叉滑移操作。具体来说，根据当前节点与第一个邻居节点之间的向量关系在Burgers向量方向上的投影vec3dotb，计算新的当前节点的位置nodep，使其与邻居节点n2在新的滑移平面上对齐。然后根据当前节点的力在滑移方向上的分量fdotglide，计算一个临时值tmp，并将其乘以滑移方向来调整当前节点的位置，以确保它在新的滑移平面上具有正确的位移。
                    nodep += tmp * glideDirLab[fplane];//如果邻居节点n1没有被固定（pinned1为false），则说明它可以被移动，因此继续进行交叉滑移操作。具体来说，根据当前节点与第一个邻居节点之间的向量关系在Burgers向量方向上的投影vec3dotb，计算新的当前节点的位置nodep，使其与邻居节点n2在新的滑移平面上对齐。然后根据当前节点的力在滑移方向上的分量fdotglide，计算一个临时值tmp，并将其乘以滑移方向来调整当前节点的位置，以确保它在新的滑移平面上具有正确的位移。
                    
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
                    splitnet.nconn = 2;//在临时的、局部的SplitNet实例中，通过移动节点位置来计算新的配置下的力，以判断是否满足交叉滑移事件的条件。具体来说，覆盖当前节点i和邻居节点n1，并设置它们的新位置nodep和nbr1p，然后计算当前节点在新的配置下的力newforce，并将其在滑移方向上的分量newfdotglide与之前计算的fdotglide进行比较，以判断当前节点是否会继续在新的滑移平面上移动。如果它们的符号相反，说明当前节点不会继续在新的滑移平面上移动，因此跳过交叉滑移事件。
                    
                    splitnet.nodes[0].pos = nodep;
                    splitnet.nodes[1].pos = nbr1p;//在临时的、局部的SplitNet实例中，覆盖当前节点i和邻居节点n1，并设置它们的新位置nodep和nbr1p，以便计算当前节点在新的配置下的力，并判断是否满足交叉滑移事件的条件。
                    
                    Vec3 newforce = force->node_force(system, &splitnet, i, team);
                    double newfdotglide = dot(newforce, glideDirLab[fplane]);//在临时的、局部的SplitNet实例中，通过移动节点位置来计算新的配置下的力newforce，并将其在滑移方向上的分量newfdotglide(试探性新配置)与之前计算的fdotglide（原始配置）进行比较，以判断当前节点是否会继续在新的滑移平面上移动。如果它们的符号相反，说明当前节点不会继续在新的滑移平面上移动，因此直接返回，不进行交叉滑移事件的处理。
                    
                    if ((SIGN(newfdotglide) * SIGN(fdotglide)) < 0.0) {
                        return;//如果当前节点在新的配置下的力在滑移方向上的分量newfdotglide与之前计算的fdotglide的符号相反，说明当前节点不会继续在新的滑移平面上移动，因此直接返回，不进行交叉滑移事件的处理。
                    }
                    
                    // Save the new node positions and plane
                    if (tid == 0) {
                        events(lid).type = 1;
                        events(lid).p0 = nodep;
                        events(lid).p1 = nbr1p;
                        events(lid).plane = newplane;
                    }//如果当前节点在新的配置下的力在滑移方向上的分量newfdotglide与之前计算的fdotglide的符号相同，说明当前节点会继续在新的滑移平面上移动，因此保存新的节点位置和新的滑移平面，以便后续的处理交叉滑移事件时使用这些信息来确定新的滑移方向和处理方式。
                }
            
            } else if (seg1_is_screw && (plane1 != plane2) && (plane2 == fplane) &&
                       (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane1])+fnodeThreshold))) {
                
                // Zipper condition met for first segment.  If the first
                // neighbor is either not pinned or pinned but already
                // sufficiently aligned, proceed with the cross-slip event
                
                bool pinned1 = node_pinned(system, net, n1, plane1, glideDirLab, numGlideDir);//如果第一个段落接近于螺旋形态，并且满足交叉滑移的条件（plane1 != plane2，plane2 == fplane，以及当前节点的力在滑移方向上的分量与plane1对应的分量之间的关系满足一定的条件），则说明第一个段落满足交叉滑移的条件。此时，需要进一步判断邻居节点n1是否被认为是固定的（pinned1为true）。如果邻居节点n1没有被固定，或者虽然被固定但已经足够对齐，那么就继续进行交叉滑移事件的处理。
                
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
                    double f1dotplane1 = fabs(dot(newSegForce, glideDirLab[plane1]));//在满足交叉滑移条件的情况下，在临时的、局部的SplitNet实例中，通过创建一个新的节点来计算新的配置下的力newSegForce，并将其在plane1对应的滑移方向上的分量f1dotplane1与一个基于系统参数和之前计算的长度L1的阈值zipperThreshold进行比较，以判断是否满足交叉滑移事件的条件。如果newSegForce在plane1对应的滑移方向上的分量f1dotplane1小于zipperThreshold加上之前计算的fnodeThreshold，那么说明当前配置下的力不足以支持交叉滑移事件，因此直接返回，不进行交叉滑移事件的处理。
                    double f1dotplanef = fabs(dot(newSegForce, newplane));
                    
                    if (f1dotplanef < zipperThreshold + f1dotplane1) {
                        return;
                    }//在满足交叉滑移条件的情况下，在临时的、局部的SplitNet实例中，通过创建一个新的节点来计算新的配置下的力newSegForce，并将其在plane1对应的滑移方向上的分量f1dotplane1与一个基于系统参数和之前计算的长度L1的阈值zipperThreshold进行比较，以判断是否满足交叉滑移事件的条件。如果newSegForce在plane1对应的滑移方向上的分量f1dotplane1小于zipperThreshold加上之前计算的fnodeThreshold，那么说明当前配置下的力不足以支持交叉滑移事件，因此直接返回，不进行交叉滑移事件的处理。
                    
                    if (!pinned1) {
                        double vec2dotb = dot(vec2, burg);
                        nbr1p = nodep - vec2dotb * burg;
                    }//如果邻居节点n1没有被固定（pinned1为false），则说明它可以被移动，因此继续进行交叉滑移操作。具体来说，根据当前节点与第一个邻居节点之间的向量关系在Burgers向量方向上的投影vec2dotb，计算新的邻居节点n1的位置nbr1p，使其与当前节点在新的滑移平面上对齐。
                    
                    // Save the new node position and plane
                    if (tid == 0) {
                        events(lid).type = 2;
                        events(lid).p1 = nbr1p;
                        events(lid).plane = newplane;
                    }//如果邻居节点n1没有被固定（pinned1为false），则说明它可以被移动，因此继续进行交叉滑移操作。具体来说，根据当前节点与第一个邻居节点之间的向量关系在Burgers向量方向上的投影vec2dotb，计算新的邻居节点n1的位置nbr1p，使其与当前节点在新的滑移平面上对齐。然后保存新的邻居节点位置和新的滑移平面，以便后续的处理交叉滑移事件时使用这些信息来确定新的滑移方向和处理方式。
                }
                
            } else if (seg2_is_screw && (plane1 != plane2) && (plane1 == fplane) &&
                       (fabs(tmp3C[fplane]) > (weightFactor*fabs(tmp3C[plane2])+fnodeThreshold))) {
                
                // Zipper condition met for second segment
                
                bool pinned2 = node_pinned(system, net, n2, plane2, glideDirLab, numGlideDir);//如果第二个段落接近于螺旋形态，并且满足交叉滑移的条件（plane1 != plane2，plane1 == fplane，以及当前节点的力在滑移方向上的分量与plane2对应的分量之间的关系满足一定的条件），则说明第二个段落满足交叉滑移的条件。此时，需要进一步判断邻居节点n2是否被认为是固定的（pinned2为true）。如果邻居节点n2没有被固定，或者虽然被固定但已经足够对齐，那么就继续进行交叉滑移事件的处理。
                
                if ((!pinned2) || ((testmax2-test2) < (eps*eps*burgSize*burgSize))) {
                    
                    // Compute force on segment by creating a temporary new node
                    // within the temporary, local SplitNet instance
                    Vec3 pmid = 0.5 * (nodep + nbr2p);
                    int nnew = splitnet.split_seg(s2, pmid);
                    
                    Vec3 newSegForce = force->node_force(system, &splitnet, nnew, team);
                    
                    double zipperThreshold = noiseFactor * shearModulus *
                                             burgSize *  L2;
                    double f1dotplane2 = fabs(dot(newSegForce, glideDirLab[plane2]));
                    double f1dotplanef = fabs(dot(newSegForce, newplane));//在满足交叉滑移条件的情况下，在临时的、局部的SplitNet实例中，通过创建一个新的节点来计算新的配置下的力newSegForce，并将其在plane2对应的滑移方向上的分量f1dotplane2与一个基于系统参数和之前计算的长度L2的阈值zipperThreshold进行比较，以判断是否满足交叉滑移事件的条件。如果newSegForce在plane2对应的滑移方向上的分量f1dotplane2小于zipperThreshold加上之前计算的fnodeThreshold，那么说明当前配置下的力不足以支持交叉滑移事件，因此直接返回，不进行交叉滑移事件的处理。

                    if (f1dotplanef < zipperThreshold + f1dotplane2) {
                        return;
                    }//在满足交叉滑移条件的情况下，在临时的、局部的SplitNet实例中，通过创建一个新的节点来计算新的配置下的力newSegForce，并将其在plane2对应的滑移方向上的分量f1dotplane2与一个基于系统参数和之前计算的长度L2的阈值zipperThreshold进行比较，以判断是否满足交叉滑移事件的条件。如果newSegForce在plane2对应的滑移方向上的分量f1dotplane2小于zipperThreshold加上之前计算的fnodeThreshold，那么说明当前配置下的力不足以支持交叉滑移事件，因此直接返回，不进行交叉滑移事件的处理。
                    
                    if (!pinned2) {
                        double vec3dotb = dot(vec3, burg);
                        nbr2p = nodep - vec3dotb * burg;
                    }//如果邻居节点n2没有被固定（pinned2为false），则说明它可以被移动，因此继续进行交叉滑移操作。具体来说，根据当前节点与第一个邻居节点之间的向量关系在Burgers向量方向上的投影vec3dotb，计算新的邻居节点n2的位置nbr2p，使其与当前节点在新的滑移平面上对齐。
                    
                    // Save the new node position and plane
                    if (tid == 0) {
                        events(lid).type = 3;
                        events(lid).p1 = nbr2p;
                        events(lid).plane = newplane;
                    }//如果邻居节点n2没有被固定（pinned2为false），则说明它可以被移动，因此继续进行交叉滑移操作。具体来说，根据当前节点与第一个邻居节点之间的向量关系在Burgers向量方向上的投影vec3dotb，计算新的邻居节点n2的位置nbr2p，使其与当前节点在新的滑移平面上对齐。然后保存新的邻居节点位置和新的滑移平面，以便后续的处理交叉滑移事件时使用这些信息来确定新的滑移方向和处理方式。
                }
            }
        }
    };
    
    void handle(System* system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_CROSSSLIP].start();
        
        if (system->crystal.type != FCC_CRYSTAL && system->crystal.type != BCC_CRYSTAL)
            ExaDiS_fatal("Error: CrossSlipParallel only implemented for FCC and BCC crystals\n");//在处理交叉滑移事件时，首先进行一些基本的检查和初始化。具体来说，首先检查系统的晶体类型是否为FCC或BCC，如果不是，则输出错误信息并终止程序。然后获取当前活跃的网络，并初始化一个FindCrossSlipEvents结构体来存储交叉滑移事件的信息。接下来，识别需要考虑交叉滑移事件的节点，并根据需要构建一个邻居列表，以便在后续的处理交叉滑移事件时快速访问相关节点的信息。
        
        if (!system->crystal.use_glide_planes)
            ExaDiS_fatal("Error: CrossSlipParallel requires use_glide_planes option\n");//在处理交叉滑移事件时，首先进行一些基本的检查和初始化。具体来说，首先检查系统的晶体类型是否为FCC或BCC，如果不是，则输出错误信息并终止程序。然后获取当前活跃的网络，并初始化一个FindCrossSlipEvents结构体来存储交叉滑移事件的信息。接下来，识别需要考虑交叉滑移事件的节点，并根据需要构建一个邻居列表，以便在后续的处理交叉滑移事件时快速访问相关节点的信息。
        
        int active_net = system->net_mngr->get_active();
        DeviceDisNet* net = system->get_device_network();
        
        // Initialize the FindCrossSlipEvents structure
        FindCrossSlipEvents* cs = exadis_new<FindCrossSlipEvents>(system, net, force);
        
        // Identify nodes attached to screw segments that need
        // to be considered for a cross-slip event
        Kokkos::resize(cs->count, 1);//1代表只有一个计数器，用于统计需要考虑交叉滑移事件的节点数量。通过调用Kokkos::resize函数来调整cs->count的大小，以便在后续的处理交叉滑移事件时使用这个计数器来统计相关节点的数量。
        Kokkos::deep_copy(cs->csnodes, 0);
        Kokkos::resize(cs->csnodes, net->Nnodes_local);
        
        Kokkos::parallel_for(net->Nnodes_local, *cs);
        Kokkos::fence();
        
        int numcsnodes = cs->count(0);
        Kokkos::resize(cs->csnodes, numcsnodes);
        
        // If we need a neighbor list, let's build a contiguous
        // one for only the subset of split nodes so that access 
        // on device will be much faster
        double cutoff = system->neighbor_cutoff;//如果需要构建邻居列表，那么我们将为只有需要考虑交叉滑移事件的节点子集构建一个连续的邻居列表，以便在设备上更快地访问相关节点的信息。具体来说，我们首先获取系统的邻居列表的截断距离cutoff，如果cutoff大于0.0，则说明需要构建邻居列表。然后创建一个NeighborBox对象来构建一个针对段落的邻居列表，并使用这个NeighborBox对象来构建一个针对节点的邻居列表，最后将这个邻居列表存储在FindCrossSlipEvents结构体中，以便在后续的处理交叉滑移事件时快速访问相关节点的信息。
        NeighborList* neilist;
        if (cutoff > 0.0) {
            NeighborBox* neighbox = exadis_new<NeighborBox>(system, cutoff, Neighbor::NeiSeg);
            // Build a neighbor list of the nodes wrt to the segs
            neilist = neighbox->build_neighbor_list(system, net, Neighbor::NeiNode, cs->csnodes);
            cs->neilist = neilist;
            exadis_delete(neighbox);
        }//如果需要构建邻居列表，那么我们将为只有需要考虑交叉滑移事件的节点子集构建一个连续的邻居列表，以便在设备上更快地访问相关节点的信息。具体来说，我们首先获取系统的邻居列表的截断距离cutoff，如果cutoff大于0.0，则说明需要构建邻居列表。然后创建一个NeighborBox对象来构建一个针对段落的邻居列表，并使用这个NeighborBox对象来构建一个针对节点的邻居列表，最后将这个邻居列表存储在FindCrossSlipEvents结构体中，以便在后续的处理交叉滑移事件时快速访问相关节点的信息。
        
        // Find all cross-slip events that we need to handle.
        // This is done in parallel where each node previously
        // identified is now assigned a team of threads.
        Kokkos::resize(cs->events, numcsnodes);//找到所有需要处理的交叉滑移事件。这个过程是并行进行的，每个之前识别的节点现在被分配给一个线程团队。具体来说，我们调整cs->events的大小以存储所有需要处理的交叉滑移事件的信息，然后使用Kokkos::parallel_for函数来并行地执行FindCrossSlipEvents结构体中的operator()函数，以便在设备上快速地识别和存储所有需要处理的交叉滑移事件的信息。
        Kokkos::parallel_for(Kokkos::TeamPolicy<>(numcsnodes, Kokkos::AUTO), *cs);
        Kokkos::fence();
        
        // We are done with determining the cross-slip events, now
        // execute the changes. We do this in serial for simplicity.
        auto h_events = Kokkos::create_mirror_view(cs->events);//我们已经完成了确定交叉滑移事件的过程，现在开始执行这些事件的更改。为了简单起见，我们以串行的方式执行这些更改。具体来说，我们首先创建一个cs->events的镜像视图h_events，以便在主机上访问和修改交叉滑移事件的信息。然后我们将当前活跃的网络设置为系统的网络管理器，并获取系统的串行网络对象，以便在后续的处理交叉滑移事件时使用这个网络对象来访问和修改相关节点的信息。
        auto h_csnodes = Kokkos::create_mirror_view(cs->csnodes);
        
        // We did not make any changes to the network yet, so
        // let's avoid making unnecessary memory copies
        system->net_mngr->set_active(active_net);
        SerialDisNet* network = system->get_serial_network();//我们已经完成了确定交叉滑移事件的过程，现在开始执行这些事件的更改。为了简单起见，我们以串行的方式执行这些更改。具体来说，我们首先创建一个cs->events的镜像视图h_events，以便在主机上访问和修改交叉滑移事件的信息。然后我们将当前活跃的网络设置为系统的网络管理器，并获取系统的串行网络对象，以便在后续的处理交叉滑移事件时使用这个网络对象来访问和修改相关节点的信息。
        
        std::vector<int> eventflag(numcsnodes, 0);
        // -1: skip, 0: not executed, 1: done
        for (int k = 0; k < numcsnodes; k++) {
            if (h_events(k).type < 0) eventflag[k] = -1;
        }//-1 0 1分别表示跳过、未执行和已完成。我们首先创建一个大小为numcsnodes的整数向量eventflag，并将其初始化为0。然后我们遍历所有的交叉滑移事件，如果某个事件的类型小于0，则将对应的eventflag设置为-1，表示这个事件需要被跳过，不进行处理。
        
        // Start with the zipper events
        for (int k = 0; k < numcsnodes; k++) {
            if (eventflag[k] != 0) continue;
            
            CrossSlipEvent& event = h_events(k);
            int type = event.type;//首先处理zipper事件。我们遍历所有的交叉滑移事件，如果某个事件的eventflag不等于0，则跳过这个事件，继续处理下一个事件。对于每个需要处理的事件，我们获取它的类型type，并检查它是否是zipper事件（type为2或3）。如果不是zipper事件，则继续处理下一个事件。
            if (type < 2) continue; // zipper events = {2,3}
            
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
                if (eventflag[l] != 1) continue;//如果当前zipper事件影响了一个之前已经参与过zipper事件的节点，我们将只允许当前事件继续进行，如果两个事件的滑移平面匹配。具体来说，我们遍历之前的所有事件，如果某个事件的eventflag不等于1，则跳过这个事件，继续处理下一个事件。对于每个之前的事件，我们获取它的类型prev_type，并检查它是否是zipper事件（prev_type为2或3）。如果不是zipper事件，则继续处理下一个事件。
                
                CrossSlipEvent& prev_event = h_events(l);
                int prev_type = prev_event.type;
                if (prev_type < 2) continue; // zipper events = {2,3}
                
                int prev_i = h_csnodes(l); // node id
                int prev_n = network->conn[prev_i].node[prev_type-2]; // neighbor id
                
                if ((i == prev_n) || (n == prev_i) || (n == prev_n)) {
                    if (cross(prev_event.plane, newplane).norm2() > 1.0e-3) {
                        skip = 1;
                        break;
                    }//如果当前zipper事件影响了一个之前已经参与过zipper事件的节点，我们将只允许当前事件继续进行，如果两个事件的滑移平面匹配。具体来说，我们遍历之前的所有事件，如果某个事件的eventflag不等于1，则跳过这个事件，继续处理下一个事件。对于每个之前的事件，我们获取它的类型prev_type，并检查它是否是zipper事件（prev_type为2或3）。如果是zipper事件，我们获取它的节点id prev_i 和邻居id prev_n，并检查当前事件的节点id i 和邻居id n 是否与之前事件的节点id prev_i 和邻居id prev_n 有交集。如果有交集，我们计算当前事件的滑移平面newplane与之前事件的滑移平面prev_event.plane的叉积，并检查其范数是否大于一个小阈值（1.0e-3）。如果大于这个阈值，说明两个滑移平面不匹配，因此将skip设置为1，并跳出循环。
                }
            }
            if (skip) continue;
            eventflag[k] = 1;
            
            // Reposition neighbor node
            network->move_node(n, event.p1, system->dEp);
            // Update the segment glide plane
            update_seg_plane(network, s, newplane);
        }
        
        // Continue with the cross-slip events
        for (int k = 0; k < numcsnodes; k++) {
            if (eventflag[k] != 0) continue;//继续处理cross-slip事件。我们遍历所有的交叉滑移事件，如果某个事件的eventflag不等于0，则跳过这个事件，继续处理下一个事件。对于每个需要处理的事件，我们获取它的类型type，并检查它是否是cross-slip事件（type为0或1）。如果不是cross-slip事件，则继续处理下一个事件。
            
            CrossSlipEvent& event = h_events(k);
            int type = event.type;
            if (type > 1) continue; // cross-slip events = {0,1}
            
            int i = h_csnodes(k); // node id
            int n = network->conn[i].node[1-type]; // neighbor id
            int n1 = network->conn[i].node[0];
            int n2 = network->conn[i].node[1];
            int s1 = network->conn[i].seg[0]; // seg 1 id
            int s2 = network->conn[i].seg[1]; // seg 2 id
            Vec3 newplane = event.plane;//继续处理cross-slip事件。我们遍历所有的交叉滑移事件，如果某个事件的eventflag不等于0，则跳过这个事件，继续处理下一个事件。对于每个需要处理的事件，我们获取它的类型type，并检查它是否是cross-slip事件（type为0或1）。如果是cross-slip事件，我们获取它的节点id i 和邻居id n，以及当前节点连接的两个邻居节点n1和n2，以及当前节点连接的两个段落s1和s2，并获取当前事件的新滑移平面newplane。
            
            // If the current cross-slip event affects a node that was
            // involved in a previous cross-slip event, we will only allow
            // the current event to proceeed if the glide planes
            // for the two events match.
            bool skip = 0;
            for (int l = 0; l < k; l++) {
                if (eventflag[l] != 1) continue;//如果当前cross-slip事件影响了一个之前已经参与过cross-slip事件的节点，我们将只允许当前事件继续进行，如果两个事件的滑移平面匹配。具体来说，我们遍历之前的所有事件，如果某个事件的eventflag不等于1，则跳过这个事件，继续处理下一个事件。对于每个之前的事件，我们获取它的类型prev_type，并检查它是否是cross-slip事件（prev_type为0或1）。如果不是cross-slip事件，则继续处理下一个事件。
                
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
                    }//如果当前cross-slip事件影响了一个之前已经参与过cross-slip事件的节点，我们将只允许当前事件继续进行，如果两个事件的滑移平面匹配。具体来说，我们遍历之前的所有事件，如果某个事件的eventflag不等于1，则跳过这个事件，继续处理下一个事件。对于每个之前的事件，我们获取它的类型prev_type，并检查它是否是cross-slip事件（prev_type为0或1）。如果是zipper事件，我们获取它的节点id prev_i 和邻居id prev_n，并检查当前事件的节点id i 和邻居id n1、n2 是否与之前事件的节点id prev_i 和邻居id prev_n 有交集。如果有交集，我们计算当前事件的滑移平面newplane与之前事件的滑移平面prev_event.plane的叉积，并检查其范数是否大于一个小阈值（1.0e-3）。如果大于这个阈值，说明两个滑移平面不匹配，因此将skip设置为1，并跳出循环。
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
                    }
                }
            }//如果当前cross-slip事件影响了一个之前已经参与过cross-slip事件的节点，我们将只允许当前事件继续进行，如果两个事件的滑移平面匹配。具体来说，我们遍历之前的所有事件，如果某个事件的eventflag不等于1，则跳过这个事件，继续处理下一个事件。对于每个之前的事件，我们获取它的类型prev_type，并检查它是否是cross-slip事件（prev_type为0或1）。如果是cross-slip事件，我们获取它的节点id prev_i 和邻居id prev_n1、prev_n2，并检查当前事件的节点id i 和邻居id n1、n2 是否与之前事件的节点id prev_i 和邻居id prev_n1、prev_n2 有交集。如果有交集，我们计算当前事件的滑移平面newplane与之前事件的滑移平面prev_event.plane的叉积，并检查其范数是否大于一个小阈值（1.0e-3）。如果大于这个阈值，说明两个滑移平面不匹配，因此将skip设置为1，并跳出循环。
            if (skip) continue;
            eventflag[k] = 1;
            
            // Reposition nodes
            network->move_node(i, event.p0, system->dEp);
            network->move_node(n, event.p1, system->dEp);
            // Update segments glide plane
            update_seg_plane(network, s1, newplane);
            update_seg_plane(network, s2, newplane);
        }
        
        
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

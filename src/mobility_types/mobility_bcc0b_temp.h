/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *  
 *  Junjie
 *-------------------------------------------------------------------------*/

#pragma once//预处理指令，防止头文件被多次包含
#ifndef EXADIS_MOBILITY_BCC0B_temp_H//如果没有定义EXADIS_MOBILITY_BCC0B_temp_H，则继续执行下面的代码
#define EXADIS_MOBILITY_BCC0B_temp_H//定义EXADIS_MOBILITY_BCC0B_temp_H，防止头文件被多次包含

#include "mobility.h"//包含mobility.h头文件，提供MobilityLocal类的定义

namespace ExaDiS {//定义ExaDiS命名空间

/*---------------------------------------------------------------------------
 *
 *    Class:        MobilityBCC0b_temp
 *
 *-------------------------------------------------------------------------*/
struct MobilityBCC0b_temp//定义MobilityBCC0b_temp结构体，表示BCC晶体的移动性模型
{
    bool non_linear = false;//标记是否启用非线性迁移率（摩擦应力非0时启用，默认线性）
    double Beclimbj;//攀移拖拽系数的结点临时值
    double Bscrew2, Beclimb2;//screw和climb拖拽系数的平方值
    double Bline, BlmBsc, BlmBecl;//Bline：位错结线的拖拽系数 BlmBsc = Bline - Bscrew、BlmBecl = Bline - Beclimbj
    double invBscrew2, invBedge2;//screw和edge拖拽系数的倒数的平方值
    double Bedge, Bscrew, Beclimb;//edge、screw和climb的拖拽系数
    double vmax, vscale;//vmax：速度上限 vscale：速度缩放因子（将m/s转换为模拟单位）
    double Fedge, Fscrew;//edge和screw的摩擦应力
    double kT, bT;//kT和bT：温度与pstrain的线性关系参数，T = kT * pstrain + bT
    
    struct Params {//MobilityBCC0b_temp的参数结构体，包含攀移拖拽系数、温度关系参数、摩擦应力和速度上限等参数
        double Mclimb;//攀移迁移率：迁移率 = 1 / 拖拽系数（Beclimb=1/Mclimb），表征位错攀移能力。
        double kT, bT; // Junjie: the slope and intercept for T vs pstrain
        double Fedge = 0.0, Fscrew = 0.0;//摩擦应力
        double vmax = -1.0;//速度上限，默认值为-1.0表示不启用速度上限
        Params() { Mclimb = vmax = -1.0; }//默认构造函数，初始化Mclimb和vmax为-1.0
        Params(double _Mclimb) {
            Mclimb = _Mclimb;
            vmax = -1.0;
        }//构造函数，接受攀移迁移率参数，默认速度上限为-1.0（不启用）
        Params(double _Mclimb, double _vmax) {
            Mclimb = _Mclimb;
            vmax = _vmax;
        }//构造函数，接受攀移迁移率和速度上限参数
        Params(double _Mclimb, double _kT, double _bT,
               double _Fedge, double _Fscrew, double _vmax) {
            Mclimb = _Mclimb;
            kT = _kT;
            bT = _bT;
            Fedge = _Fedge;
            Fscrew = _Fscrew;
            vmax = _vmax;
        }//构造函数，接受攀移迁移率、温度关系参数、摩擦应力和速度上限等参数
    };
    
    MobilityBCC0b_temp(System* system, Params& params)//构造函数，接受系统指针和参数结构体，初始化MobilityBCC0b_temp对象
    {
        if (system->crystal.type != BCC_CRYSTAL)
            ExaDiS_fatal("Error: MobilityBCC0b() must be used with BCC crystal type\n");//检查系统的晶体类型是否为BCC，如果不是则输出错误信息并终止程序
            
        if (params.Mclimb < 0.0 || params.Fedge < 0 || params.Fscrew < 0.0)
            ExaDiS_fatal("Error: invalid MobilityBCC0b() parameter values\n");//检查攀移迁移率和摩擦应力参数是否有效，如果无效则输出错误信息并终止程序
        
        Beclimb = 1.0 / params.Mclimb;//根据攀移迁移率计算攀移拖拽系数
        vmax    = params.vmax;
        vscale  = system->params.burgmag; //vscale (convert factor from m/s)相当于位错弓长的单位长度（通常为b），将速度从m/s转换为模拟单位
        Fedge   = params.Fedge;
        Fscrew  = params.Fscrew;
        kT      = params.kT;
        bT      = params.bT;
        
        if (Fedge > 1e-5 || Fscrew > 1e-5)
            non_linear = true;//如果edge或screw的摩擦应力大于1e-5，则启用非线性迁移率
        
        // Initialization
        Beclimbj   = Beclimb;//攀移拖拽系数的结点临时值初始化为Beclimb
        Beclimb2   = Beclimb * Beclimb;//攀移拖拽系数的平方值
    }
    
    KOKKOS_INLINE_FUNCTION//内联函数，计算screw拖拽系数Bscrew作为pstrain（温度）和tau_rss的函数
    double compute_Bscrew(double pstrain, double tau_rss)//计算screw拖拽系数Bscrew作为pstrain（温度）和tau_rss的函数
    {
        // Junjie: Helper function to compute Bscrew as a function of pstrain (temperature) and tau_rss 
        // Adjustable parameters
        double A = 1.8804e-5;//A是一个可调参数，表示screw拖拽系数的基准值
        double dH0 = 0.257; // eV, activation enthalpy at 0 K, using Ti66Nb17Zr17 as an approximation "F. Maresca and W.A. Curtin, Acta Materialia 182 (2020) 144–162"
        double p = 0.5614;
        double q = 0.5987;
        double tau_p = 814.91e6; // extrapolated Peierls stress in Pa 通过实验外推得到的Peierls应力，单位为Pa

        // Derived parameters
        double Tm = 1589 + 273.15; // Melting temperature Tm = 1589 C per the experiment
        double T = kT * pstrain + bT; // Temperature as a function of pstrain 温度作为pstrain的函数，线性关系由kT和bT定义
        double kB = 8.617333262145e-5; // eV/K, Boltzmann constant 波尔兹曼常数，单位为eV/K
        if (tau_rss >= tau_p) // 如果tau_rss大于或等于Peierls应力，则screw拖拽系数Bscrew直接与温度成正比，表示在高应力下screw位错的移动能力增强
        {
            return A * T;//screw拖拽系数Bscrew直接与温度成正比，表示在高应力下screw位错的移动能力增强
        } else {
            double dG = dH0 * (pow((1 - pow(tau_rss / tau_p, p)),q)  - T / Tm);
            if (dG < 0.0) dG = 0.0;//计算screw位错的活化能dG，考虑了tau_rss对活化能的影响以及温度的影响。如果dG小于0，则将其设置为0，表示在高温或高应力下screw位错的移动能力增强
            return A * T * exp(dG / (kB * T));
        }
    }

    KOKKOS_INLINE_FUNCTION
    double compute_Bedge(double pstrain)//计算edge拖拽系数Bedge作为pstrain（温度）的函数
    {
        // Junjie: Helper function to compute Bedge as a function of pstrain (temperature)
        // Adjustable parameters
        double A = 1.47002720e-04;//从1.8804e-5调整为1.47002720e-04（在gpu上面跑跑试一试）

        // Derived parameters
        double Tm = 1589 + 273.15; // Melting temperature Tm = 1589 C per the experiment
        double T = kT * pstrain + bT; // Temperature as a function of pstrain
        return A*T;
    }

    template<class N>//模板函数，计算节点i的速度，根据系统状态、网络结构和作用力fi
    KOKKOS_INLINE_FUNCTION
    Vec3 node_velocity(System *system, N *net, const int &i, const Vec3 &fi)//计算节点i的速度，根据系统状态、网络结构和作用力fi
    {
        // Determine Bedge wrt. the current pstrain (temperature)
        double pstrain = system->pstrain;//从系统状态中获取当前的pstrain（温度）
        Bedge = compute_Bedge(pstrain);//根据当前的pstrain计算edge拖拽系数Bedge
        invBedge2  = 1.0 / (Bedge*Bedge);//计算edge拖拽系数Bedge的倒数的平方值，供后续计算使用
        
        auto nodes = net->get_nodes();//从网络结构中获取节点信息
        auto segs = net->get_segs();//从网络结构中获取段信息
        auto conn = net->get_conn();//从网络结构中获取连接信息
        auto cell = net->cell;//从网络结构中获取晶胞信息
        
        Vec3 vi(0.0);//初始化节点i的速度vi为0
        
        if (conn[i].num >= 2 && nodes[i].constraint != PINNED_NODE) {//如果节点i连接的段数大于或等于2，并且节点i没有被固定约束，则计算节点i的速度
            
            int linejunc = 0;//linejunc标记是否为结线节点，初始值为0
            Vec3 tjunc(0.0);//tjunc表示结线方向的单位向量，初始值为0
            
            if (system->params.split3node) {//如果系统参数中启用了split3node功能，则检查节点i是否为二元结点或平面结点，并设置linejunc和tjunc的值
                
                int binaryjunc = 0;//binaryjunc标记是否为二元结点，初始值为0
                int planarjunc = 0;//planarjunc标记是否为平面结点，初始值为0
                binaryjunc = BCC_binary_junction_node(system, net, i, tjunc, &planarjunc);  //检查节点i是否为二元结点，并计算结线方向的单位向量tjunc。如果节点i是二元结点，则binaryjunc为1；如果节点i是平面结点，则planarjunc为1。              
                binaryjunc = (binaryjunc > -1);//将binaryjunc转换为布尔值，表示节点i是否为二元结点

                if (binaryjunc) {//如果节点i是二元结点，则linejunc标记为1，表示该节点是结线节点
                    if (planarjunc) {
                        linejunc = 1;
                    } else {//如果节点i是二元结点但不是平面结点，则根据作用力fi和结线方向tjunc的点积来判断节点i是否正在解开结线。如果点积大于0，则表示节点i正在解开结线，此时将linejunc标记为1，表示该节点是结线节点
                        int unzipping = (dot(fi, tjunc) > 0.0);
                        // If the node is unzipping the junction and we are
                        // not treating unzipping as a purely topological
                        // operation, then orthogonalize the climb direction
                        if (unzipping) linejunc = 1;//如果节点i正在解开结线，并且我们不将解开结线视为纯粹的拓扑操作，则将linejunc标记为1，表示该节点是结线节点
                    }
                }
            }

            double eps = 1e-12;//eps是一个小的数值，用于避免除以零或处理非常小的数值时的数值不稳定问题
            Mat33 Btotal = Mat33().zero();//初始化总拖拽矩阵Btotal为零矩阵
            double FricForce = 0.0;//初始化摩擦力FricForce为0

            // Build drag matrix
            Vec3 r1 = nodes[i].pos;//获取节点i的位置r1
            int numNonZeroLenSegs = 0;//numNonZeroLenSegs用于统计连接到节点i的段中长度非零的段的数量，初始值为0
            for (int j = 0; j < conn[i].num; j++) {//循环遍历连接到节点i的每个段，计算每个段对节点i的拖拽贡献，并累加到总拖拽矩阵Btotal中

                int k = conn[i].node[j];//获取连接到节点i的第j个段的另一个节点k
                int s = conn[i].seg[j];//获取连接到节点i的第j个段的段索引s
                int order = conn[i].order[j];//获取连接到节点i的第j个段的位错类型order，通常为+1或-1，表示位错的方向

                Vec3 burg = order*segs[s].burg;//根据段s的位错矢量burg和段的位错类型order计算段s的有效位错矢量burg，考虑了位错的方向
                double bMag = burg.norm();//计算段s的有效位错矢量burg的模长bMag，表示段s的位错强度
                double bMag2 = bMag*bMag;
                double invbMag2 = 1.0 / bMag2;//计算段s的有效位错矢量burg的模长的平方bMag2，以及模长的平方的倒数invbMag2，供后续计算使用

                Vec3 r2 = cell.pbc_position(r1, nodes[k].pos);//根据节点i的位置r1和连接到节点i的另一个节点k的位置nodes[k].pos，考虑周期性边界条件，计算节点k在周期性边界条件下的位置r2
                
                Vec3 dr = r2-r1;//计算节点i和节点k之间的位移向量dr，表示段s的方向和长度
                double mag = dr.norm();//计算位移向量dr的模长mag，表示段s的长度
                if (mag < eps) continue;
                numNonZeroLenSegs++;//如果段s的长度mag小于eps，则跳过该段的计算；否则，统计连接到节点i的长度非零的段的数量numNonZeroLenSegs加1

                double halfMag = 0.5 * mag;
                double invMag  = 1.0 / mag;
                dr = invMag * dr;//将位移向量dr归一化，得到段s的单位方向向量dr

                double costheta = dot(dr, burg);//计算段s的单位方向向量dr与段s的有效位错矢量burg的点积costheta，表示段s的方向与位错矢量的夹角余弦值
                double costheta2 = (costheta*costheta) * invbMag2;//计算costheta的平方值costheta2，并乘以invbMag2，得到costheta2的归一化值，供后续计算使用
                
                double dangle = 1.0 / bMag * fabs(costheta);//计算dangle，表示段s的方向与位错矢量的夹角的函数，考虑了位错强度bMag的影响。dangle越大，表示段s的方向与位错矢量越接近，摩擦应力越接近Fscrew；dangle越小，表示段s的方向与位错矢量越垂直，摩擦应力越接近Fedge。
                double fricStress = Fedge+(Fscrew-Fedge)*dangle;//根据dangle计算段s的摩擦应力fricStress，表示段s的摩擦应力在Fedge和Fscrew之间，根据段s的方向与位错矢量的夹角进行插值
                FricForce += fricStress * bMag * mag;//根据段s的摩擦应力fricStress、位错强度bMag和段长度mag计算段s的摩擦力，并累加到总摩擦力FricForce中

                Bscrew = compute_Bscrew(pstrain, fi.norm() / bMag / mag);//根据当前的pstrain和作用力fi的大小计算screw拖拽系数Bscrew，考虑了作用力对screw位错移动能力的影响
                Bscrew2    = Bscrew * Bscrew;
                Bline      = 1.0e-2 * MIN(Bscrew, Bedge);//根据screw拖拽系数Bscrew和edge拖拽系数Bedge计算位错结线的拖拽系数Bline，取两者的较小值，并乘以1.0e-2作为比例因子
                BlmBsc     = Bline - Bscrew;
                BlmBecl    = Bline - Beclimbj;
                invBscrew2 = 1.0 / (Bscrew*Bscrew);//计算screw拖拽系数Bscrew的平方的倒数invBscrew2，供后续计算使用
                
                if (bMag > 1.0+eps) {
                    // [0 0 1] arms don't move as readily as other arms, so must be
                    // handled specially.
                    
                    // The junction node move along the junction line freely
                    // No drag on the junction line
                    if (linejunc == 1) {
                        Btotal[0][0] += halfMag * Bline;//对于[0 0 1]方向的段，由于其移动能力较差，因此需要特殊处理。如果节点i是结线节点，则沿结线方向的拖拽系数为Bline，表示结线节点在结线方向上可以自由移动，没有拖拽。
                        Btotal[1][1] += halfMag * Bline;
                        Btotal[2][2] += halfMag * Bline;
                        continue;
                    }
                    
                    if (conn[i].num == 2) {
                        Btotal[0][0] += halfMag * Beclimbj;//如果连接到节点i的段数为2，则认为该节点是一个二元结点，此时沿段s的方向的拖拽系数为Beclimbj，表示二元结点在段s的方向上只能通过攀移来移动，而不能通过滑移来移动。
                        Btotal[1][1] += halfMag * Beclimbj;
                        Btotal[2][2] += halfMag * Beclimbj;
                    } else {
                        Btotal[0][0] += halfMag * (dr.x*dr.x * BlmBecl + Beclimbj);//如果连接到节点i的段数大于2，则认为该节点是一个平面结点，此时沿段s的方向的拖拽系数为BlmBecl + Beclimbj，表示平面结点在段s的方向上既有攀移的拖拽又有结线的拖拽；而在垂直于段s的方向上的拖拽系数为Beclimbj，表示平面结点在垂直于段s的方向上只能通过攀移来移动。
                        Btotal[0][1] += halfMag * (dr.x*dr.y * BlmBecl);
                        Btotal[0][2] += halfMag * (dr.x*dr.z * BlmBecl);
                        Btotal[1][1] += halfMag * (dr.y*dr.y * BlmBecl + Beclimbj);
                        Btotal[1][2] += halfMag * (dr.y*dr.z * BlmBecl);
                        Btotal[2][2] += halfMag * (dr.z*dr.z * BlmBecl + Beclimbj);
                    }
                } else  {
                    // Arm is not [0 0 1], so build the drag matrix assuming the
                    // dislocation is screw type
                    Btotal[0][0] += halfMag * (dr.x*dr.x * BlmBsc + Bscrew);//对于非[0 0 1]方向的段，构建拖拽矩阵时假设位错为screw类型，因此沿段s的方向的拖拽系数为BlmBsc + Bscrew，表示screw位错在段s的方向上既有结线的拖拽又有screw的拖拽；而在垂直于段s的方向上的拖拽系数为Bscrew，表示screw位错在垂直于段s的方向上只能通过滑移来移动。
                    Btotal[0][1] += halfMag * (dr.x*dr.y * BlmBsc);
                    Btotal[0][2] += halfMag * (dr.x*dr.z * BlmBsc);
                    Btotal[1][1] += halfMag * (dr.y*dr.y * BlmBsc + Bscrew);
                    Btotal[1][2] += halfMag * (dr.y*dr.z * BlmBsc);
                    Btotal[2][2] += halfMag * (dr.z*dr.z * BlmBsc + Bscrew);

                    // Now correct the drag matrix for dislocations that are
                    // not screw
                    if ((1.0 - costheta2) > eps) {//如果段s的方向与位错矢量的夹角的余弦值的平方costheta2与1的差值大于eps，则说明段s的方向与位错矢量不完全平行，此时需要对拖拽矩阵进行修正，考虑段s的方向与位错矢量的夹角对拖拽系数的影响。

                        double invsqrt1mcostheta2 = 1.0 / sqrt((1.0 - costheta2) * bMag2);//计算1.0 / sqrt((1.0 - costheta2) * bMag2)，表示段s的方向与位错矢量的夹角对拖拽系数的修正因子，考虑了位错强度bMag的影响
                        Vec3 nr = cross(burg, dr);//计算nr，表示段s的方向与位错矢量的叉积，表示段s的方向与位错矢量的垂直方向
                        nr = invsqrt1mcostheta2 * nr;//将nr乘以修正因子invsqrt1mcostheta2，得到nr的修正值，表示段s的方向与位错矢量的夹角对拖拽系数的修正后的垂直方向

                        Vec3 mr = cross(nr, dr);//计算mr，表示nr与段s的单位方向向量dr的叉积，表示段s的方向与位错矢量的夹角的另一个垂直方向
                        
                        // Orthogonalize climb direction wrt junction line direction
                        // to avoid numerical issues with binary junction nodes
                        if (linejunc == 1) {//如果节点i是结线节点，则将nr正交化到结线方向tjunc上，以避免二元结点的数值问题
                            nr -= dot(nr, tjunc) * tjunc;//将nr减去其在结线方向tjunc上的分量，得到nr在结线方向上的正交分量
                            nr = nr.normalized();//将nr归一化，得到nr的单位向量，表示段s的方向与位错矢量的夹角的垂直方向在结线方向上的正交分量的单位向量
                        }
                        
                        double Bglide = sqrt(invBedge2+(invBscrew2-invBedge2)*costheta2);//计算Bglide，表示段s的方向与位错矢量的夹角对滑移拖拽系数的修正，考虑了edge和screw拖拽系数的影响以及段s的方向与位错矢量的夹角的影响
                        Bglide = 1.0 / Bglide;//将Bglide取倒数，得到滑移拖拽系数的修正值，表示段s的方向与位错矢量的夹角对滑移拖拽系数的修正后的值
                        double Bclimb = sqrt(Beclimb2 + (Bscrew2 - Beclimb2) * costheta2);//计算Bclimb，表示段s的方向与位错矢量的夹角对攀移拖拽系数的修正，考虑了攀移和screw拖拽系数的影响以及段s的方向与位错矢量的夹角的影响
                        double BclmBsc = Bclimb - Bscrew;
                        double BglmBsc = Bglide - Bscrew;

                        Btotal[0][0] += halfMag * (nr.x*nr.x * BclmBsc +
                                                   mr.x*mr.x * BglmBsc);//根据nr和mr的分量以及BclmBsc和BglmBsc的值，计算段s的方向与位错矢量的夹角对拖拽矩阵的修正，并累加到总拖拽矩阵Btotal中
                        Btotal[0][1] += halfMag * (nr.x*nr.y * BclmBsc +
                                                   mr.x*mr.y * BglmBsc);
                        Btotal[0][2] += halfMag * (nr.x*nr.z * BclmBsc +
                                                   mr.x*mr.z * BglmBsc);
                        Btotal[1][1] += halfMag * (nr.y*nr.y * BclmBsc +
                                                   mr.y*mr.y * BglmBsc);
                        Btotal[1][2] += halfMag * (nr.y*nr.z * BclmBsc +
                                                   mr.y*mr.z * BglmBsc);
                        Btotal[2][2] += halfMag * (nr.z*nr.z * BclmBsc +
                                                   mr.z*mr.z * BglmBsc);
                    }
                }  // End non-[0 0 1] arm
            }  // End loop over arms

            Btotal[1][0] = Btotal[0][1];//由于拖拽矩阵Btotal是对称的，因此将Btotal的非对角元素进行赋值，使其满足对称性
            Btotal[2][0] = Btotal[0][2];
            Btotal[2][1] = Btotal[1][2];
            FricForce /= 2.0;//将总摩擦力FricForce除以2，得到平均摩擦力，考虑了每个段对节点i的拖拽贡献被计算了两次（一次是从节点i到节点k，另一次是从节点k到节点i）

            if (numNonZeroLenSegs > 0) {//如果连接到节点i的长度非零的段的数量numNonZeroLenSegs大于0，则计算节点i的速度。否则，节点i的速度保持为0。
                Mat33 invDragMatrix = Btotal.inverse();//计算总拖拽矩阵Btotal的逆矩阵invDragMatrix，供后续计算使用
                
                Vec3 f = fi;//将作用力fi赋值给f，表示节点i的总作用力
                if (FricForce > eps) {//如果总摩擦力FricForce大于eps，则对作用力f进行修正，考虑摩擦力的影响。具体来说，如果作用力f的大小fmag大于总摩擦力FricForce，则将f减去一个与f方向相同、大小为FricForce的向量；否则，将f设置为0，表示作用力被完全抵消。
                    double fmag = f.norm();//计算作用力f的大小fmag
                    if (fmag > FricForce) {
                        f -= FricForce/fmag * f;
                    } else {
                        f = Vec3(0.0);//如果作用力f的大小fmag不大于总摩擦力FricForce，则将f设置为0，表示作用力被完全抵消
                    }
                }
                
                vi = invDragMatrix * f;//根据总拖拽矩阵的逆矩阵invDragMatrix和修正后的作用力f计算节点i的速度vi，考虑了拖拽矩阵和摩擦力的影响
                if (vmax > 0.0) //如果速度上限vmax大于0，则对节点i的速度vi进行限制，确保其不超过速度上限vmax。具体来说，如果vi的大小超过vmax，则将vi缩放到vmax的大小，保持其方向不变。
                    apply_velocity_cap(vmax, vscale, vi);
            }
        }
        
        return vi;
    }
    
    static constexpr const char* name = "MobilityBCC0b_temp";
};

namespace MobilityType {
    typedef MobilityLocal<MobilityBCC0b_temp> BCC_0B_temp;//在MobilityType命名空间中定义BCC_0B_temp类型，表示使用MobilityBCC0b_temp模型的局部移动性类型
}

} // namespace ExaDiS

#endif

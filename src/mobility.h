/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once//预处理指令，防止头文件被多次包含导致编译错误
#ifndef EXADIS_MOBILITY_H//如果没有定义EXADIS_MOBILITY_H，则继续执行下面的代码
#define EXADIS_MOBILITY_H

#include "system.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        Mobility
 *
 *-------------------------------------------------------------------------*/
class Mobility {
public:
    bool non_linear = false;//标记是否启用非线性迁移率，默认为 false，表示使用线性迁移率模型，以便在后续的模拟过程中能够根据用户指定的条件来控制迁移率模型，并且确保在每个模拟步骤中都有一个完整的迁移率记录
    Mobility() {}
    Mobility(System *system) {}
    virtual void compute(System *system) = 0;//纯虚函数，计算节点的速度，根据系统状态来计算每个节点的速度，以便在后续的模拟过程中能够正确地更新节点的位置和状态，并且确保在每个模拟步骤中都有一个完整的速度记录
    virtual Vec3 node_velocity(System *system, const int &i, const Vec3 &fi) = 0;//纯虚函数，计算节点i的速度，根据系统状态和作用力fi来计算节点i的速度，以便在后续的模拟过程中能够正确地更新节点的位置和状态，并且确保在每个模拟步骤中都有一个完整的速度记录
    virtual ~Mobility() {}
    virtual const char* name() { return "MobilityNone"; }//虚函数，返回移动性模型的名称，默认为 "MobilityNone"，表示没有启用任何移动性模型，以便在后续的模拟过程中能够根据用户指定的条件来控制移动性模型，并且确保在每个模拟步骤中都有一个完整的移动性记录
};

/*---------------------------------------------------------------------------
 *
 *    Class:        MobilityLocal
 *                  Base class for local types of mobilities in which the
 *                  node velocity is computed by looping over its arm and
 *                  summing a drag contribution.
 *
 *-------------------------------------------------------------------------*/
template <class M>//MobilityLocal类模板，表示局部移动性类型的基类，其中M是具体的移动性模型类型，MobilityLocal类提供了计算节点速度的基本框架，通过循环遍历节点的连接臂并累加拖拽贡献来计算节点的速度，以便在后续的模拟过程中能够正确地更新节点的位置和状态，并且确保在每个模拟步骤中都有一个完整的速度记录
class MobilityLocal : public Mobility {
public:
    typedef M Mob;//定义Mob类型别名，表示具体的移动性模型类型，以便在后续的模拟过程中能够根据用户指定的条件来控制移动性模型，并且确保在每个模拟步骤中都有一个完整的移动性记录
    typedef typename M::Params Params;
    M *mob; // mobility kernel
    
    MobilityLocal(System *system, Params params) {
        mob = exadis_new<M>(system, params);
        non_linear = mob->non_linear;
    }//构造函数，接受系统指针和参数结构体，初始化MobilityLocal对象，并根据具体的移动性模型来设置非线性迁移率标记，以便在后续的模拟过程中能够根据用户指定的条件来控制迁移率模型，并且确保在每个模拟步骤中都有一个完整的迁移率记录
    
    template<class N>
    struct NodeMobility {
        System *system;
        M *mob;
        N *net;
        NodeMobility(System *_system, M *_mob, N *_net) : system(_system), mob(_mob), net(_net) {}//构造函数，接受系统指针、移动性模型指针和网络指针，初始化NodeMobility对象，以便在后续的模拟过程中能够正确地计算节点的速度，并且确保在每个模拟步骤中都有一个完整的速度记录
        
        KOKKOS_INLINE_FUNCTION
        void operator()(const int &i) const {
            auto nodes = net->get_nodes();
            nodes[i].v = mob->node_velocity(system, net, i, nodes[i].f);//重载函数调用运算符，接受节点索引i，根据系统状态、网络结构和节点的作用力来计算节点i的速度，并将计算结果存储在节点的速度属性中，以便在后续的模拟过程中能够正确地更新节点的位置和状态，并且确保在每个模拟步骤中都有一个完整的速度记录
        }
    };
    
    void compute(System *system)
    {
        Kokkos::fence();
        system->timer[system->TIMER_MOBILITY].start();//计算节点的速度，根据系统状态来计算每个节点的速度，以便在后续的模拟过程中能够正确地更新节点的位置和状态，并且确保在每个模拟步骤中都有一个完整的速度记录
        
        DeviceDisNet *net = system->get_device_network();//从系统状态中获取设备网络对象，以便在后续的模拟过程中能够正确地计算节点的速度，并且确保在每个模拟步骤中都有一个完整的速度记录
        using policy = Kokkos::RangePolicy<Kokkos::LaunchBounds<32,1>>;//定义Kokkos的并行策略，使用RangePolicy来指定并行范围，并且使用LaunchBounds来优化线程块的大小和数量，以便在后续的模拟过程中能够正确地计算节点的速度，并且确保在每个模拟步骤中都有一个完整的速度记录
        Kokkos::parallel_for(policy(0, net->Nnodes_local), NodeMobility<DeviceDisNet>(system, mob, net));//使用Kokkos的parallel_for来并行计算节点的速度，遍历设备网络中的每个本地节点，并调用NodeMobility的函数调用运算符来计算每个节点的速度，以便在后续的模拟过程中能够正确地更新节点的位置和状态，并且确保在每个模拟步骤中都有一个完整的速度记录
        
        Kokkos::fence();//等待所有并行计算完成，以确保在后续的模拟过程中能够正确地计算节点的速度，并且确保在每个模拟步骤中都有一个完整的速度记录
        system->timer[system->TIMER_MOBILITY].stop();//停止计时器，记录计算节点速度的时间，以便在后续的模拟过程中能够监控模拟的性能和效率，并且确保在每个模拟步骤中都有一个完整的时间记录
    }
    
    Vec3 node_velocity(System *system, const int &i, const Vec3 &fi)
    {
        SerialDisNet *network = system->get_serial_network();
        return mob->node_velocity(system, network, i, fi);
    }//计算节点i的速度，根据系统状态、网络结构和作用力fi来计算节点i的速度，以便在后续的模拟过程中能够正确地更新节点的位置和状态，并且确保在每个模拟步骤中都有一个完整的速度记录
    
    ~MobilityLocal() {
        exadis_delete(mob);
    }//析构函数，释放移动性模型对象的内存，以便在后续的模拟过程中能够正确地管理内存资源，并且确保在每个模拟步骤中都有一个完整的内存记录
    
    const char* name() { return M::name; }//返回移动性模型的名称，调用具体移动性模型的静态成员name来获取名称，以便在后续的模拟过程中能够根据用户指定的条件来控制移动性模型，并且确保在每个模拟步骤中都有一个完整的移动性记录
};

/*---------------------------------------------------------------------------
 *
 *    Function:     apply_velocity_cap
 *
 *-------------------------------------------------------------------------*/
KOKKOS_FORCEINLINE_FUNCTION
void apply_velocity_cap(const double &vmax, const double &vscale, Vec3 &v)
{
    if (vmax <= 0.0) return;
    double vmag = v.norm() * vscale; // m/s
    if (vmag < 1e-5) return;
    double alpha = 10.0;//alpha是一个调节参数，用于控制速度上限的平滑程度，较大的alpha值会使速度上限更接近于vmax，而较小的alpha值会使速度上限更平滑地过渡到vmax，以便在后续的模拟过程中能够正确地限制节点的速度，并且确保在每个模拟步骤中都有一个完整的速度记录
    double vcap = vmag / pow(1.0 + pow(vmag/vmax, alpha), 1.0/alpha);//根据当前的速度大小vmag和速度上限vmax来计算实际的速度上限vcap，使用一个平滑函数来限制速度，使其在接近vmax时逐渐减小，以便在后续的模拟过程中能够正确地限制节点的速度，并且确保在每个模拟步骤中都有一个完整的速度记录
    v = (vcap / vmag) * v;//将节点的速度v缩放到实际的速度上限vcap，以便在后续的模拟过程中能够正确地限制节点的速度，并且确保在每个模拟步骤中都有一个完整的速度记录
}

} // namespace ExaDiS


// Available mobility types
#include "mobility_glide.h"
#include "mobility_bcc0b.h"
#include "mobility_fcc0.h"
#include "mobility_fcc0_fric.h"
#include "mobility_fcc0b.h"
#include "mobility_bcc_nl.h"
#include "mobility_bcc0b_temp.h" // Junjie

#endif

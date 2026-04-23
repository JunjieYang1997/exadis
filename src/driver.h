/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_DRIVER_H//头文件保护，防止重复包含导致编译错误
#define EXADIS_DRIVER_H//exadis（离散位错动力学模拟框架）的核心驱动头文件

#include "exadis.h"//包含 exadis 的核心头文件，提供系统、模块、工具函数等的定义和声明
#include <iostream>//包含输入输出流库，用于打印日志、错误信息等

namespace ExaDiS {//封装所有核心逻辑避免命名冲突

/*---------------------------------------------------------------------------
 *
 *    Class:        ExaDiSApp
 *                  Example of a driver to peform a DDD simulation
 *
 *-------------------------------------------------------------------------*/
class ExaDiSApp {//离散位错动力学模拟应用类，负责管理模拟的整体流程，包括系统和模块的设置、模拟的初始化、每个模拟步骤的执行以及模拟结果的输出等，以确保模拟能够按照用户指定的条件正确地进行，并且能够监控模拟的进展和性能
public:
    System* system = nullptr;//系统对象，存储位错网络、材料参数、应力 / 应变等，是模拟的核心数据容器
    Force* force = nullptr;
    Mobility* mobility = nullptr;
    Integrator* integrator = nullptr;
    Collision* collision = nullptr;
    Topology* topology = nullptr;
    Remesh* remesh = nullptr;//模拟模块对象，分别负责计算力、移动性、时间积分、碰撞处理、拓扑变化处理和重网格化等，是模拟的核心计算模块
    CrossSlip* crossslip = nullptr;
    std::string outputdir = "";//输出目录路径，存储模拟结果、日志文件等，以便在后续的模拟过程中能够正确地保存和组织输出数据
    
    bool dealloc = true;//标记析构时是否释放内存，默认为 true，表示在析构时会释放 system 和各个模块的内存，以确保资源的正确管理和避免内存泄漏
    bool setup = false;//标记模拟是否已经完成基础配置，默认为 false，在模拟初始化完成后会设置为 true，以便在后续的模拟过程中能够正确地进行第一次迭代的初始化操作，确保模拟能够按照用户指定的条件正确地进行迭代
    bool init = false;//标记模拟是否已经完成初始化，默认为 false，在模拟初始化完成后会设置为 true，以便在后续的模拟过程中能够正确地进行第一次迭代的初始化操作，确保模拟能够按照用户指定的条件正确地进行迭代
    bool log = true;//标记是否启用日志记录，默认为 true，表示在模拟过程中会记录日志信息，以便在后续的模拟过程中能够监控模拟的进展和分析模拟结果，并且确保在每个模拟步骤中都有一个完整的输出记录
    bool restart = false;//标记是否从重启文件读取模拟状态，默认为 false，在模拟初始化完成后如果指定了重启文件则会设置为 true，以便在后续的模拟过程中不再进行重启文件的读取操作，确保模拟能够按照用户指定的条件正确地进行迭代
    
    int istep;//当前模拟步数，记录模拟的迭代次数，以便在后续的模拟过程中能够正确地反映当前的模拟步数，确保模拟能够按照用户指定的条件正确地进行迭代
    Mat33 Etot;//总应变张量，记录系统的总应变状态，以便在后续的模拟过程中能够正确地反映当前的应变状态，确保模拟能够按照用户指定的条件正确地进行迭代
    double stress, strain, pstrain;//当前的应力、应变和塑性应变，记录系统的当前力学状态，以便在后续的模拟过程中能够正确地反映当前的力学状态，确保模拟能够按照用户指定的条件正确地进行迭代
    double tottime;//当前模拟总时间，记录模拟的总时间，以便在后续的模拟过程中能够正确地反映当前的时间状态，确保模拟能够按照用户指定的条件正确地进行迭代
    Vec3 edir;//当前应变方向，记录系统的当前应变方向，以便在后续的模拟过程中能够正确地反映当前的应变方向，确保模拟能够按照用户指定的条件正确地进行迭代
    
    Kokkos::Timer timer;//计时器对象，用于测量模拟的运行时间，以便在后续的模拟过程中能够监控模拟的性能和效率，并且确保在每个模拟步骤中都有一个完整的时间记录
    bool timeronefile = true;//标记是否将时间记录写入单独的文件，默认为 true，表示在模拟过程中会将时间记录写入一个单独的文件，以便在后续的模拟过程中能够正确地保存和组织时间数据，并且确保在每个模拟步骤中都有一个完整的时间记录
    double outfiletime = 0.0;//记录上一次输出的时间，以便在后续的模拟过程中能够根据控制参数中的输出频率 outfreqdt 来决定是否进行输出操作，确保模拟能够按照用户指定的条件正确地进行迭代
    int outfilecounter = 0;//记录输出文件的计数器，以便在后续的模拟过程中能够根据控制参数中的输出频率 outfreq 来决定是否进行输出操作，并且确保每个输出文件都有一个唯一的编号，确保模拟能够按照用户指定的条件正确地进行迭代
    
    struct Stepper {
        enum Types {NUM_STEPS, MAX_STEPS, MAX_STRAIN, MAX_TIME, MAX_WALLTIME};//迭代器类型，定义了不同的停止条件类型，包括基于步数、应变、时间或壁钟时间的停止条件，以便在模拟过程中能够根据用户指定的条件来控制迭代过程
        int type = NUM_STEPS;
        int maxsteps = 0;
        double stopval = 0.0;//停止值，记录基于应变、时间或壁钟时间的停止条件的具体数值，以便在模拟过程中能够根据用户指定的条件来控制迭代过程，并且确保在每个模拟步骤中都有一个完整的停止条件记录
        Stepper(int _type, int _maxsteps) : type(_type), maxsteps(_maxsteps) {}
        Stepper(int _type, double _stopval) : type(_type), stopval(_stopval) {}//构造函数，根据不同的停止条件类型来初始化 Stepper 对象，确保在模拟过程中能够根据用户指定的条件来控制迭代过程，并且确保在每个模拟步骤中都有一个完整的停止条件记录
        Stepper& operator=(int nsteps) { type = NUM_STEPS; maxsteps = nsteps; return *this; }//赋值操作符重载，允许直接将一个整数赋值给 Stepper 对象来设置基于步数的停止条件，确保在模拟过程中能够根据用户指定的条件来控制迭代过程，并且确保在每个模拟步骤中都有一个完整的停止条件记录
        bool iterate(ExaDiSApp* exadis);//迭代函数，根据当前的停止条件类型和模拟状态来判断是否继续迭代，并在第一次调用时进行相应的初始化，以确保模拟能够按照用户指定的条件正确地进行迭代
    };
    static Stepper NUM_STEPS(int nsteps) { return Stepper(Stepper::NUM_STEPS, nsteps); }//静态函数，创建一个 Stepper 对象，设置基于步数的停止条件，确保在模拟过程中能够根据用户指定的条件来控制迭代过程，并且确保在每个模拟步骤中都有一个完整的停止条件记录
    static Stepper MAX_STEPS(int maxsteps) { return Stepper(Stepper::MAX_STEPS, maxsteps); }
    static Stepper MAX_STRAIN(double maxstrain) { return Stepper(Stepper::MAX_STRAIN, maxstrain); }
    static Stepper MAX_TIME(double maxtime) { return Stepper(Stepper::MAX_TIME, maxtime); }
    static Stepper MAX_WALLTIME(double maxtime) { return Stepper(Stepper::MAX_WALLTIME, maxtime); }
    
    struct Prop {
        enum fields {STEP, STRAIN, STRESS, DENSITY, NNODES, NSEGS, DT, TIME, WALLTIME, EDIR, RORIENT, ALLSTRESS};//输出字段枚举，定义了不同的输出字段类型，包括步数、应变、应力、密度、节点数、线段数、时间步长、总时间、壁钟时间、应变方向、旋转矩阵和所有应力分量等，以便在模拟过程中能够根据用户指定的条件来控制输出内容，并且确保在每个模拟步骤中都有一个完整的输出记录
        static fields get_field(std::string& name) {
            if (name == "Step" || name == "step") return STEP;//静态函数，根据输入的字段名称来返回对应的枚举值，以便在模拟过程中能够根据用户指定的条件来控制输出内容，并且确保在每个模拟步骤中都有一个完整的输出记录
            else if (name == "Strain" || name == "strain") return STRAIN;//如果输入的字段名称是 "Strain" 或 "strain"，则返回 STRAIN 枚举值，以便在模拟过程中能够根据用户指定的条件来控制输出内容，并且确保在每个模拟步骤中都有一个完整的输出记录
            else if (name == "Stress" || name == "stress") return STRESS;
            else if (name == "Density" || name == "density") return DENSITY;
            else if (name == "Nnodes") return NNODES;
            else if (name == "Nsegs") return NSEGS;
            else if (name == "DT" || name == "dt") return DT;
            else if (name == "Time" || name == "time") return TIME;
            else if (name == "Walltime" || name == "walltime") return WALLTIME;
            else if (name == "edir") return EDIR;
            else if (name == "Rorient") return RORIENT;//旋转矩阵字段，表示输出当前的晶体旋转状态，以便在模拟过程中能够根据用户指定的条件来控制输出内容，并且确保在每个模拟步骤中都有一个完整的输出记录
            else if (name == "Allstress" || name == "allstress") return ALLSTRESS;//所有应力分量字段，表示输出当前的外部应力状态，以便在模拟过程中能够根据用户指定的条件来控制输出内容，并且确保在每个模拟步骤中都有一个完整的输出记录
            else ExaDiS_fatal("Unknown control property name = %s\n", name.c_str());
            return STEP;
        }//如果输入的字段名称不匹配任何已定义的字段，则输出错误信息并终止程序，以确保在模拟过程中能够根据用户指定的条件来控制输出内容，并且确保在每个模拟步骤中都有一个完整的输出记录
    };
    
    enum Loadings {STRESS_CONTROL, STRAIN_RATE_CONTROL};//加载类型枚举，定义了不同的加载方式，包括应力控制和应变率控制，以便在模拟过程中能够根据用户指定的条件来控制加载方式，并且确保在每个模拟步骤中都有一个完整的加载记录
    struct Control {
        Stepper nsteps = NUM_STEPS(100);
        int loading = STRESS_CONTROL;//加载方式，默认为应力控制，以便在模拟过程中能够根据用户指定的条件来控制加载方式，并且确保在每个模拟步骤中都有一个完整的加载记录
        double erate = 1e3;//应变率
        Vec3 edir = Vec3(0.0, 0.0, 1.0);//加载方向
        Mat33 appstress = Mat33().zero();//应用的应力张量，默认为零张量，以便在模拟过程中能够根据用户指定的条件来控制加载方式，并且确保在每个模拟步骤中都有一个完整的加载记录
        int rotation = 0;//加载旋转，表示在加载过程中是否进行旋转，以便在模拟过程中能够根据用户指定的条件来控制加载方式，并且确保在每个模拟步骤中都有一个完整的加载记录
        int printfreq = 1;//控制台输出频率，默认为每步输出一次，以便在模拟过程中能够根据用户指定的条件来控制输出频率，并且确保在每个模拟步骤中都有一个完整的输出记录
        int propfreq = 10;//属性输出频率，默认为每 10 步输出一次，以便在模拟过程中能够根据用户指定的条件来控制输出频率，并且确保在每个模拟步骤中都有一个完整的输出记录
        int outfreq = 100;//输出频率，默认为每 100 步输出一次，以便在模拟过程中能够根据用户指定的条件来控制输出频率，并且确保在每个模拟步骤中都有一个完整的输出记录
        double outfreqdt = -1.0;//基于时间的输出频率，默认为 -1.0，表示不启用基于时间的输出频率，以便在模拟过程中能够根据用户指定的条件来控制输出频率，并且确保在每个模拟步骤中都有一个完整的输出记录
        int oprecwritefreq = 0;//操作记录输出频率，默认为 0，表示不启用操作记录输出，以便在模拟过程中能够根据用户指定的条件来控制操作记录的输出频率，并且确保在每个模拟步骤中都有一个完整的输出记录
        int oprecfilefreq = 0;//操作记录文件频率，默认为 0，表示不启用操作记录文件的递增，以便在模拟过程中能够根据用户指定的条件来控制操作记录文件的输出频率，并且确保每个操作记录文件都有一个唯一的编号，确保模拟能够按照用户指定的条件正确地进行迭代
        int oprecposfreq = 0;//操作记录位置输出频率，默认为 0，表示不启用操作记录位置的输出，以便在模拟过程中能够根据用户指定的条件来控制操作记录位置的输出频率，并且确保在每个模拟步骤中都有一个完整的输出记录
        std::vector<Prop::fields> props = {Prop::STEP, Prop::STRAIN, Prop::STRESS, Prop::DENSITY};//属性字段列表，默认为步数、应变、应力和密度，以便在模拟过程中能够根据用户指定的条件来控制输出内容，并且确保在每个模拟步骤中都有一个完整的输出记录
        void set_props(std::vector<std::string> fields) {
            props.clear();//清空当前的属性字段列表，以便在后续的模拟过程中能够根据用户指定的条件来控制输出内容，并且确保在每个模拟步骤中都有一个完整的输出记录
            for (auto f : fields)//循环检查输入的字段名称列表，根据每个字段名称来获取对应的枚举值，并将其添加到属性字段列表中，以便在模拟过程中能够根据用户指定的条件来控制输出内容，并且确保在每个模拟步骤中都有一个完整的输出记录
                props.push_back(Prop::get_field(f));
        }//根据输入的字段名称列表，循环检查每个字段名称，并将其对应的枚举值添加到属性字段列表中，以便在模拟过程中能够根据用户指定的条件来控制输出内容，并且确保在每个模拟步骤中都有一个完整的输出记录
    };
    
    ExaDiSApp(int argc, char* argv[]);//构造函数，根据命令行参数来初始化 ExaDiSApp 对象，包括解析控制参数、设置输出目录等，以便在后续的模拟过程中能够正确地进行初始化操作，并且确保模拟能够按照用户指定的条件正确地进行迭代
    ExaDiSApp();
    ~ExaDiSApp();
    
    virtual void set_modules(
        Force* _force,
        Mobility* _mobility,
        Integrator* _integrator,
        Collision* _collision,
        Topology* _topology,
        Remesh* _remesh,
        CrossSlip* _crossslip = nullptr
    );//设置模拟模块的函数，根据输入的模块对象来设置 ExaDiSApp 对象的模块成员变量，以便在后续的模拟过程中能够正确地调用这些模块进行相应的计算和处理，并且确保模拟能够按照用户指定的条件正确地进行迭代
    virtual void set_simulation(std::string restartfile="");//设置模拟的函数，根据输入的重启文件来初始化模拟状态，如果指定了重启文件则从中读取模拟状态，否则进行常规的模拟初始化，以便在后续的模拟过程中能够正确地进行迭代，并且确保模拟能够按照用户指定的条件正确地进行迭代
    virtual void set_directory();//设置输出目录的函数，根据当前的日期和时间来创建一个新的输出目录，以便在后续的模拟过程中能够正确地保存和组织输出数据，并且确保每次模拟都有一个唯一的输出目录，确保模拟能够按照用户指定的条件正确地进行迭代
    
    virtual void initialize(Control& ctrl, bool check_modules=true);//初始化模拟的函数，根据输入的控制参数来进行模拟的初始化操作，包括检查模块是否设置、设置输出目录、从重启文件读取模拟状态（如果指定了重启文件）以及进行常规的模拟初始化等，以便在后续的模拟过程中能够正确地进行迭代，并且确保模拟能够按照用户指定的条件正确地进行迭代
    virtual void step(Control& ctrl);//执行一个模拟步骤的函数，根据输入的控制参数来执行一个模拟步骤，包括更新力学状态、计算力、处理碰撞、进行时间积分、处理拓扑变化、进行重网格化以及进行交滑移等，以便在后续的模拟过程中能够正确地进行迭代，并且确保模拟能够按照用户指定的条件正确地进行迭代
    virtual void run(Control& ctrl);//运行模拟的函数，根据输入的控制参数来运行整个模拟过程，包括在每个模拟步骤中执行相应的操作，并且根据停止条件来控制迭代过程，以便在后续的模拟过程中能够正确地进行迭代，并且确保模拟能够按照用户指定的条件正确地进行迭代
    virtual void update_mechanics(Control& ctrl);//更新力学状态的函数，根据当前的加载控制方式（应变率控制或应力控制）进行相应的计算和更新，包括旋转应变方向和外部应力、计算应变增量和应力增量、更新总应变和总塑性应变、以及更新总时间等，以确保模拟的力学状态与当前加载条件一致，并且在启用操作记录时添加相应的操作记录以便未来分析
    virtual void output(Control& ctrl);//输出当前的应力、应变、配置等信息，根据控制参数中的输出频率和输出字段进行相应的输出操作，包括在控制台打印信息、写入属性数据文件、输出配置文件和重启文件，以及记录计时器信息等，以便用户能够监控模拟的进展和分析模拟结果
    
    virtual void oprec_save_integration(Control& ctrl);//保存积分操作记录的函数，根据输入的控制参数来决定是否保存积分操作记录，并且根据操作记录输出频率来控制操作记录的输出，以便在后续的模拟过程中能够正确地保存和组织操作记录数据，并且确保每个操作记录文件都有一个唯一的编号，确保模拟能够按照用户指定的条件正确地进行迭代
    virtual void oprec_replay(Control& ctrl, std::string oprec_file);//重放操作记录的函数，根据输入的控制参数和操作记录文件来重放之前保存的操作记录，以便在后续的模拟过程中能够正确地回放之前的模拟操作，并且确保模拟能够按照用户指定的条件正确地进行迭代
    
    virtual void write_restart(std::string restartfile);//写入重启文件的函数，根据输入的重启文件路径来将当前的模拟状态写入一个重启文件，以便在未来的模拟中能够从该重启文件中读取模拟状态并继续进行模拟，确保模拟能够按照用户指定的条件正确地进行迭代
    virtual void read_restart(std::string restartfile);//读取重启文件的函数，根据输入的重启文件路径来从一个重启文件中读取模拟状态，以便在未来的模拟中能够从该重启文件中继续进行模拟，确保模拟能够按照用户指定的条件正确地进行迭代
    
    double von_mises(const Mat33 &T) {
        Mat33 Tdev = T - 1.0/3.0*T.trace()*Mat33().eye();
        return sqrt(3.0/2.0*dot(Tdev, Tdev));
    }//计算 von Mises 应力的函数，根据输入的应力张量来计算并返回对应的 von Mises 应力值，以便在后续的模拟过程中能够正确地反映当前的应力状态，并且确保在每个模拟步骤中都有一个完整的应力记录
};
    
} // namespace ExaDiS

#endif

/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "exadis.h"
#include "driver.h"

namespace ExaDiS {//封装所有核心逻辑避免命名冲突

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp
 *                  Initialization of a DDD simulation
 *
 *-------------------------------------------------------------------------*/
ExaDiSApp::ExaDiSApp(int argc, char* argv[])
{
    // Allocate system on unified host/device space
    system = exadis_new<System>();//在主机 / 设备统一内存空间分配System对象（System是模拟的核心数据容器，存储位错网络、材料参数、应力 / 应变等）；
    
    istep = 0;//当前步数
    Etot = Mat33().zero();//总应变
    stress = strain = pstrain = 0.0;//应力、应变、塑性应变
    tottime = 0.0;//初始化模拟总时间
   
    if (system->proc_rank == 0) {
        printf("----------------------------------------------------\n");
        printf("ExaDiS\n");
        printf("----------------------------------------------------\n");
        printf("Version: %s\n", EXADIS_VERSION);//打印exadis版本
#ifdef MPI
        printf("MPI build\n");//条件编译，区分 MPI / 串行构建
#else
        printf("Serial build\n");
#endif
        Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);
    }//打印 Kokkos 执行空间配置（CPU/GPU 等），Kokkos 是异构编程框架，支撑 ExaDiS 的跨架构运行。
}

ExaDiSApp::ExaDiSApp()
{
    istep = 0;
    Etot = Mat33().zero();
    stress = strain = pstrain = 0.0;
    tottime = 0.0;
    dealloc = false;//标记析构时不释放内存（因未分配system）
}//功能：轻量化构造（无System分配），用于特殊场景（如仅读取重启文件不运行模拟）；

/*---------------------------------------------------------------------------
 *
 *    Function:     ~ExaDiSApp
 *                  Termination of a DDD simulation / free memory
 *
 *-------------------------------------------------------------------------*/
ExaDiSApp::~ExaDiSApp()
{
    if (!dealloc) return;
    exadis_delete(system);//释放 System 对象内存
    if (force) delete force;//释放力模块内存
    if (mobility) delete mobility;
    if (integrator) delete integrator;//释放积分模块内存
    if (crossslip) delete crossslip;//交滑移模块
    if (collision) delete collision;
    if (topology) delete topology;
    if (remesh) delete remesh;//释放重网格模块内存
}//逐个释放模拟模块（force/mobility等）：这些是位错动力学的核心计算模块，析构时清理。

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::set_modules()
 *                  Set the various base modules (force, mobility, etc.)
 *                  required to run a simulation
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::set_modules(
    Force* _force,
    Mobility* _mobility,
    Integrator* _integrator,
    Collision* _collision,
    Topology* _topology,
    Remesh* _remesh,
    CrossSlip* _crossslip)//注入模拟所需的核心模块（依赖注入模式），每个模块对应位错动力学的一个物理过程 仅赋值指针，模块的生命周期由外部管理（析构时释放）。
{
    force = _force;
    mobility = _mobility;
    integrator = _integrator;
    collision = _collision;
    topology = _topology;
    remesh = _remesh;
    crossslip = _crossslip;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::set_simulation()
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::set_simulation(std::string restartfile)//设置模拟参数，读取重启文件（如果提供），并准备输出目录
{
    if (!restartfile.empty()) {
        read_restart(restartfile);
        restart = true;
    }//若指定重启文件，读取重启数据并标记restart=true
    
    set_directory();//设置输出目录（创建目录、设置日志文件等）
    setup = true;//标记模拟已完成基础配置
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::set_directory()
 *                  Set the output directory
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::set_directory()//创建输出目录（如果是重启则保留原目录），并设置日志文件
{
    if (system->proc_rank == 0) {
        if (!restart && !setup) remove_directory(outputdir);
        create_directory(outputdir);//创建输出目录（如果是重启则保留原目录），以存储模拟结果、日志文件等
    }
    
    // Set log file
    std::string logfile = outputdir + "/exadis.log";//日志文件路径
    if (log) flog = fopen(logfile.c_str(), "a");//打开日志文件（追加模式），以记录模拟过程中的信息、警告和错误
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::write_restart()
 *                  Write file to restart a simulation from a previous state
 *
 *-------------------------------------------------------------------------*/
#define RESTART_VERSION "1.0"//重启文件版本号，便于未来版本兼容性检查
void ExaDiSApp::write_restart(std::string restartfile)//将当前模拟状态写入重启文件，以便未来可以从该状态继续模拟
{
    FILE* fp = fopen(restartfile.c_str(), "w");//打开重启文件进行写入，如果无法打开则报错退出
    if (fp == NULL)//文件打开失败
        ExaDiS_fatal("Error: cannot open restart file %s\n", restartfile.c_str());//输出错误信息并终止程序
        
    ExaDiS_log("Writing restart file\n");//记录日志，说明正在写入重启文件
    ExaDiS_log(" Restart file: %s\n", restartfile.c_str());//记录日志，显示重启文件的路径
    
    // header
    fprintf(fp, "# ExaDiS restart file\n");//写入重启文件头部注释，说明这是一个 ExaDiS 重启文件
    fprintf(fp, "\n");//写入空行分隔头部和内容
    fprintf(fp, "version %s\n", RESTART_VERSION);//写入重启文件版本号，便于未来版本兼容性检查
    time_t tp; time(&tp);//获取当前时间戳，并将其转换为本地时间字符串，记录在重启文件中以便追踪模拟的时间信息
    char timestr[64];//将当前时间戳转换为本地时间字符串，并去掉末尾的换行符，写入重启文件中以记录模拟的时间信息
    asctime_r(localtime(&tp), timestr);
    timestr[strlen(timestr)-1] = 0;//去掉末尾的换行符
    fprintf(fp, "date_and_time %s\n", timestr);//写入当前日期和时间信息到重启文件中，以便追踪模拟的时间信息
    fprintf(fp, "\n");
    
    // driver
    fprintf(fp, "step %d\n", istep);//写入当前步数到重启文件中，以便未来从该步数继续模拟
    fprintf(fp, "Etot %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
    Etot.xx(), Etot.xy(), Etot.xz(), Etot.yx(), Etot.yy(), Etot.yz(), Etot.zx(), Etot.zy(), Etot.zz());//写入总应变张量的9个分量到重启文件中，以便未来从该应变状态继续模拟
    fprintf(fp, "stress %.17g\n", stress);
    fprintf(fp, "strain %.17g\n", strain);
    fprintf(fp, "pstrain %.17g\n", pstrain);//写入当前应力、应变和塑性应变到重启文件中，以便未来从该状态继续模拟
    fprintf(fp, "tottime %.17g\n", tottime);//写入当前模拟总时间到重启文件中，以便未来从该时间继续模拟
    fprintf(fp, "edir %.17g %.17g %.17g\n", edir.x, edir.y, edir.z);//写入当前应变方向到重启文件中，以便未来从该方向继续模拟
    if (outfilecounter > 0 || outfiletime > 0.0) {
        fprintf(fp, "outfiletime %.17g\n", outfiletime);
        fprintf(fp, "outfilecounter %d\n", outfilecounter);
    }//写入输出文件计数器和时间到重启文件中，以便未来从该计数器和时间继续模拟
    if (system->oprec) fprintf(fp, "opreccounter %d\n", system->oprec->filecounter);//如果启用了操作记录（OpRec），则写入操作记录文件计数器到重启文件中，以便未来从该计数器继续模拟
    fprintf(fp, "\n");
    
    // system
    fprintf(fp, "extstress %.17g %.17g %.17g %.17g %.17g %.17g\n",
    system->extstress.xx(), system->extstress.yy(), system->extstress.zz(),
    system->extstress.xy(), system->extstress.xz(), system->extstress.yz());//写入当前外部应力张量的6个独立分量到重启文件中，以便未来从该应力状态继续模拟
    fprintf(fp, "realdt %.17g\n", system->realdt);//写入当前真实时间步长到重启文件中，以便未来从该时间步长继续模拟
    fprintf(fp, "density %e\n", system->density);//写入当前位错密度到重启文件中，以便未来从该密度状态继续模拟
    fprintf(fp, "\n");
    
    // integrator
    if (integrator) {
        fprintf(fp, "integrator %s\n", integrator->name());
        integrator->write_restart(fp);//如果存在时间积分器模块，则写入其名称和状态到重启文件中，以便未来从该积分器状态继续模拟
        fprintf(fp, "\n");
    }
    
    // crystal
    fprintf(fp, "crystal type %d\n", system->crystal.type);//写入当前晶体类型到重启文件中，以便未来从该晶体类型继续模拟
    fprintf(fp, "crystal orientation %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",//写入当前晶体取向矩阵的9个分量到重启文件中，以便未来从该晶体取向继续模拟
    system->crystal.R.xx(), system->crystal.R.xy(), system->crystal.R.xz(),
    system->crystal.R.yx(), system->crystal.R.yy(), system->crystal.R.yz(),
    system->crystal.R.zx(), system->crystal.R.zy(), system->crystal.R.zz());//晶体取向矩阵描述了晶体坐标系相对于模拟坐标系的旋转关系，写入该矩阵的9个分量到重启文件中，以便未来从该晶体取向继续模拟
    fprintf(fp, "\n");
    
    // network
    int active_net = system->net_mngr->get_active();//获取当前活跃的位错网络索引，以便在写入重启文件后恢复该网络的活跃状态
    SerialDisNet* net = system->get_serial_network();//获取当前活跃的位错网络对象，以便写入该网络的状态到重启文件中
    
    fprintf(fp, "pbc %d %d %d\n", net->cell.xpbc, net->cell.ypbc, net->cell.zpbc);//写入当前位错网络的周期性边界条件信息到重启文件中，以便未来从该边界条件状态继续模拟
    fprintf(fp, "H %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
    net->cell.H.xx(), net->cell.H.xy(), net->cell.H.xz(),
    net->cell.H.yx(), net->cell.H.yy(), net->cell.H.yz(),
    net->cell.H.zx(), net->cell.H.zy(), net->cell.H.zz());//写入当前位错网络的晶胞矩阵的9个分量到重启文件中，以便未来从该晶胞状态继续模拟
    fprintf(fp, "origin %.17g %.17g %.17g\n", net->cell.origin.x, net->cell.origin.y, net->cell.origin.z);//写入当前位错网络的原点位置到重启文件中，以便未来从该原点状态继续模拟
    fprintf(fp, "\n");
    
    fprintf(fp, "Nnodes %d\n", system->Nnodes_total());//写入当前位错网络的节点总数到重启文件中，以便未来从该节点数量继续模拟
    for (int i = 0; i < net->number_of_nodes(); i++)//循环写入每个节点的信息（位置和约束）到重启文件中，以便未来从该节点状态继续模拟
        fprintf(fp, "%d %.17g %.17g %.17g %d\n", i, 
        net->nodes[i].pos.x, net->nodes[i].pos.y, net->nodes[i].pos.z, 
        net->nodes[i].constraint);//节点信息包括节点索引、位置坐标和约束类型，写入这些信息到重启文件中，以便未来从该节点状态继续模拟
    fprintf(fp, "\n");
        
    fprintf(fp, "Nsegs %d\n", system->Nsegs_total());//写入当前位错网络的线段总数到重启文件中，以便未来从该线段数量继续模拟
    for (int i = 0; i < net->number_of_segs(); i++)//循环写入每个线段的信息（连接的节点索引、伯格斯矢量和平面法向量）到重启文件中，以便未来从该线段状态继续模拟
        fprintf(fp, "%d %d %.17g %.17g %.17g %.17g %.17g %.17g\n", net->segs[i].n1, net->segs[i].n2,
        net->segs[i].burg.x, net->segs[i].burg.y, net->segs[i].burg.z,
        net->segs[i].plane.x, net->segs[i].plane.y, net->segs[i].plane.z);//线段信息包括连接的两个节点索引、伯格斯矢量的三个分量和平面法向量的三个分量，写入这些信息到重启文件中，以便未来从该线段状态继续模拟
    fprintf(fp, "\n");
    
    system->net_mngr->set_active(active_net);//恢复之前获取的活跃位错网络索引，以确保在写入重启文件后继续使用同一网络进行模拟
        
    fclose(fp);//关闭重启文件，完成写入操作
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::read_restart()
 *                  Read restart file to continue a simulation
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::read_restart(std::string restartfile)//从重启文件中读取模拟状态，以便继续之前的模拟
{
#ifdef MPI
    ExaDiS_fatal("read_restart() not implemented for MPI\n");//当前实现仅支持串行读取重启文件，MPI 版本尚未实现该功能，调用该函数将报错退出
#endif
    
    FILE* fp = fopen(restartfile.c_str(), "r");//打开重启文件进行读取，如果无法打开则报错退出
    if (fp == NULL)
        ExaDiS_fatal("Error: cannot open restart file %s\n", restartfile.c_str());//文件打开失败，输出错误信息并终止程序
    
    ExaDiS_log("Reading restart file\n");//记录日志，说明正在读取重启文件
    ExaDiS_log(" Restart file: %s\n", restartfile.c_str());//记录日志，显示重启文件的路径
    
    char *line = NULL;//用于存储从重启文件中读取的每一行文本的指针，初始值为 NULL，getline 函数会根据需要分配内存
    size_t len = 0;//用于存储 getline 函数分配的缓冲区大小的变量，初始值为 0，getline 函数会根据需要更新该值
    
    // Read general information
    double version = 0.0;//用于存储重启文件版本号的变量，初始值为 0.0，读取重启文件时会更新该值以检查版本兼容性
    Cell cell;
    
    while (getline(&line, &len, fp) != -1) {//循环读取重启文件的每一行文本，直到文件末尾，使用 getline 函数自动处理行缓冲区的分配和大小调整
        // header
        if (strncmp(line, "version", 7) == 0) { sscanf(line, "version %lf\n", &version); }//版本兼容性
        
        // driver
        else if (strncmp(line, "step", 4) == 0) { sscanf(line, "step %d\n", &istep); }//读取当前步数
        else if (strncmp(line, "Etot", 4) == 0) {
            sscanf(line, "Etot %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            &Etot[0][0], &Etot[0][1], &Etot[0][2],
            &Etot[1][0], &Etot[1][1], &Etot[1][2],
            &Etot[2][0], &Etot[2][1], &Etot[2][2]);
        }//读取总应变张量的9个分量
        else if (strncmp(line, "stress", 6) == 0) { sscanf(line, "stress %lf\n", &stress); }
        else if (strncmp(line, "strain", 6) == 0) { sscanf(line, "strain %lf\n", &strain); }
        else if (strncmp(line, "pstrain", 7) == 0) { sscanf(line, "pstrain %lf\n", &pstrain); }//读取当前应力、应变和塑性应变
        else if (strncmp(line, "tottime", 7) == 0) { sscanf(line, "tottime %lf\n", &tottime); }//读取当前模拟总时间
        else if (strncmp(line, "edir", 4) == 0) { sscanf(line, "edir %lf %lf %lf\n", &edir.x, &edir.y, &edir.z); }//读取当前应变方向的三个分量
        else if (strncmp(line, "outfiletime", 11) == 0) { sscanf(line, "outfiletime %lf\n", &outfiletime); }//读取输出文件时间，以便未来从该时间继续模拟
        else if (strncmp(line, "outfilecounter", 14) == 0) { sscanf(line, "outfilecounter %d\n", &outfilecounter); }//读取输出文件计数器，以便未来从该计数器继续模拟
        else if (strncmp(line, "opreccounter", 12) == 0) {
            if (system->oprec) sscanf(line, "opreccounter %d\n", &system->oprec->filecounter);
        }//如果启用了操作记录（OpRec），则读取操作记录文件计数器，以便未来从该计数器继续模拟
        
        // system
        else if (strncmp(line, "extstress", 9) == 0) {
            double sxx, syy, szz, sxy, sxz, syz;
            sscanf(line, "extstress %lf %lf %lf %lf %lf %lf\n",
            &sxx, &syy, &szz, &sxy, &sxz, &syz);//读取当前外部应力张量的6个独立分量
            system->extstress.symmetric(sxx, syy, szz, sxy, sxz, syz);
        }//将读取的外部应力分量设置到系统的外部应力张量中，使用 symmetric 函数确保张量的对称性
        else if (strncmp(line, "realdt", 6) == 0) { 
            sscanf(line, "realdt %lf\n", &system->realdt);
            system->params.nextdt = system->realdt;
        }//读取当前的真实时间步长，并设置系统参数中的下一个时间步长，以确保时间步长的一致性
        
        // integrator
        // need to save/set some integrator class members to ensure reproducibility
        else if (strncmp(line, "integrator", 10) == 0) {
            char intname[100];//用于存储从重启文件中读取的时间积分器名称的字符数组，长度为100，读取重启文件时会更新该值以确定使用哪个时间积分器模块
            sscanf(line, "integrator %s\n", intname);
            bool set = 0;//标记是否成功设置了时间积分器模块，初始值为0，读取重启文件时如果找到了匹配的时间积分器并成功设置，则将该值更新为1
            if (integrator) {
                if (strcmp(integrator->name(), intname) == 0) {
                    integrator->read_restart(fp);
                    set = 1;
                }
            }
            if (!set) ExaDiS_log("Warning: skipped resetting the time-integrator\n");//如果未找到匹配的时间积分器模块，输出警告日志，说明跳过了时间积分器的重置，这可能会影响模拟的可重复性，但不会导致程序崩溃
        }
        
        // crystal
        else if (strncmp(line, "crystal type", 12) == 0) {
            int crystal;
            sscanf(line, "crystal type %d\n", &crystal);
            if (crystal != system->crystal.type)
                ExaDiS_fatal("Error: inconsistent crystal type for restart\n");
        }//读取当前晶体类型，并检查是否与系统中设置的晶体类型一致，如果不一致则报错退出，以确保重启文件的晶体类型与当前系统设置兼容
        else if (strncmp(line, "crystal orientation", 19) == 0) {
            Mat33 R;
            sscanf(line, "crystal orientation %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            &R[0][0], &R[0][1], &R[0][2],
            &R[1][0], &R[1][1], &R[1][2],
            &R[2][0], &R[2][1], &R[2][2]);
            system->crystal.set_orientation(R);
        }//读取当前晶体取向矩阵的9个分量，并设置系统的晶体取向，以确保重启文件的晶体取向与当前系统设置兼容
        
        // cell
        else if (strncmp(line, "pbc", 3) == 0) { sscanf(line, "pbc %d %d %d\n", &cell.xpbc, &cell.ypbc, &cell.zpbc); }//读取当前位错网络的周期性边界条件信息，并设置到 cell 对象中，以便后续创建位错网络时使用这些边界条件
        else if (strncmp(line, "H", 1) == 0) {
            Mat33 H;
            sscanf(line, "H %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            &H[0][0], &H[0][1], &H[0][2],
            &H[1][0], &H[1][1], &H[1][2],
            &H[2][0], &H[2][1], &H[2][2]);
            cell.set_H(H);
        }//读取当前位错网络的晶胞矩阵的9个分量，并设置到 cell 对象中，以便后续创建位错网络时使用该晶胞矩阵
        else if (strncmp(line, "origin", 6) == 0) { sscanf(line, "origin %lf %lf %lf\n", &cell.origin.x, &cell.origin.y, &cell.origin.z); }//读取当前位错网络的原点位置，并设置到 cell 对象中，以便后续创建位错网络时使用该原点位置
        
        // network
        else if (strncmp(line, "Nnodes", 6) == 0) {
            break;//读取到节点数量行时，跳出循环，准备后续读取节点和线段信息，因为节点和线段信息的格式与之前的行不同，需要单独处理
        }
    }
    
    printf(" version = %g\n", version);//打印读取的重启文件版本号，以便确认版本兼容性
    
    // network
    SerialDisNet* net = system->get_serial_network();//获取当前活跃的位错网络对象，以便将读取的位错网络状态设置到该对象中
    net->cell = cell;//将之前读取的周期性边界条件、晶胞矩阵和原点位置设置到当前位错网络的 cell 属性中，以确保重启文件的位错网络边界条件和晶胞状态与当前系统设置兼容
    net->nodes.clear();//清空当前位错网络的节点列表，以便后续读取重启文件中的节点信息并添加到该列表中，确保重启文件的节点状态与当前位错网络兼容
    net->segs.clear();//清空当前位错网络的线段列表，以便后续读取重启文件中的线段信息并添加到该列表中，确保重启文件的线段状态与当前位错网络兼容
    net->conn.clear();//清空当前位错网络的连接信息，以便后续根据读取的节点和线段信息重新生成连接关系，确保重启文件的连接状态与当前位错网络兼容
    
    int Nnodes;
    sscanf(line, "Nnodes %d\n", &Nnodes);
    printf(" nodes = %d\n", Nnodes);//读取当前位错网络的节点数量，并打印该数量，以便确认重启文件中的节点数量信息
    for (int i = 0; i < Nnodes; i++) {
        int id, constraint;
        Vec3 pos;
        fscanf(fp, "%d %lf %lf %lf %d\n", &id, &pos.x, &pos.y, &pos.z, &constraint);//循环读取每个节点的信息，包括节点索引、位置坐标和约束类型，并将这些信息设置到当前位错网络中，以确保重启文件中的节点状态与当前位错网络兼容
        net->add_node(pos, constraint);//将读取的节点信息添加到当前位错网络的节点列表中，确保重启文件中的节点状态与当前位错网络兼容
    }
    
    int Nsegs;
    fscanf(fp, "Nsegs %d\n", &Nsegs);
    printf(" segments = %d\n", Nsegs);//读取当前位错网络的线段数量，并打印该数量，以便确认重启文件中的线段数量信息
    for (int i = 0; i < Nsegs; i++) {
        int n1, n2;
        Vec3 burg, plane;
        fscanf(fp, "%d %d %lf %lf %lf %lf %lf %lf\n", &n1, &n2,
        &burg.x, &burg.y, &burg.z, &plane.x, &plane.y, &plane.z);
        net->add_seg(n1, n2, burg, plane);//循环读取每个线段的信息，包括连接的节点索引、伯格斯矢量和平面法向量，并将这些信息设置到当前位错网络中，以确保重启文件中的线段状态与当前位错网络兼容
    }
    
    net->generate_connectivity();//根据读取的节点和线段信息重新生成当前位错网络的连接关系，以确保重启文件中的连接状态与当前位错网络兼容
    net->update_ptr();//更新当前位错网络的指针信息，以确保重启文件中的节点和线段状态与当前位错网络兼容
    system->density = net->dislocation_density(system->params.burgmag);//根据当前位错网络的状态计算位错密度，并设置到系统的 density 属性中，以确保重启文件中的位错密度状态与当前系统设置兼容
    
    free(line);//释放 getline 函数分配的行缓冲区内存，以避免内存泄漏
    fclose(fp);//关闭重启文件，完成读取操作
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::update_mechanics()
 *                  Update stress, strain, crystal rotation, etc.
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::update_mechanics(Control& ctrl)//更新力学状态，包括应力、应变、晶体旋转等，根据当前的加载控制方式（应变率控制或应力控制）进行相应的计算和更新
{
    if (ctrl.rotation) {
        // Counter-rotate stress wrt dislocation configuration
        double p1 = - system->dWp.zy();
        double p2 =   system->dWp.zx();
        double p3 = - system->dWp.yx();//根据系统的旋转速率张量 dWp 计算旋转角度 p1、p2 和 p3，这些角度用于构建旋转矩阵，以便将应变方向和外部应力从当前位错配置的坐标系旋转回全局坐标系
        
        #define Ss(a) ((a))//定义宏 Ss(a) 直接返回输入参数 a，表示旋转矩阵中的正弦项
        #define Cs(a) (1.0 - (0.5 * (a)*(a)))//定义宏 Cs(a) 计算旋转矩阵中的余弦项，使用小角度近似（cos(a) ≈ 1 - 0.5*a^2）来提高计算效率，同时保持足够的精度，因为旋转角度通常较小
        Mat33 Rspin;
        Rspin[0][0] =  Cs(p3)*Cs(p2);
        Rspin[1][1] =  Cs(p3)*Cs(p1) + Ss(p3)*Ss(p2)*Ss(p1);
        Rspin[2][2] =  Cs(p2)*Cs(p1);
        Rspin[0][1] = -Ss(p3)*Cs(p1) + Cs(p3)*Ss(p1)*Ss(p2);
        Rspin[1][2] = -Cs(p3)*Ss(p1) + Ss(p3)*Ss(p2)*Cs(p1);
        Rspin[2][0] = -Ss(p2);
        Rspin[0][2] =  Ss(p3)*Ss(p1) + Cs(p3)*Cs(p1)*Ss(p2);
        Rspin[1][0] =  Ss(p3)*Cs(p2);
        Rspin[2][1] =  Cs(p2)*Ss(p1);
        
        edir = Rspin * edir;//将当前的应变方向 edir 通过旋转矩阵 Rspin 进行旋转，以将其从当前位错配置的坐标系旋转回全局坐标系，确保后续的应变率控制计算使用正确的应变方向
        edir = edir.normalized();//将旋转后的应变方向 edir 进行归一化，以确保其长度为1，保持其作为一个纯方向向量的性质，避免由于旋转引入的数值误差导致应变方向的长度发生变化
        
        // Rotate stress
        system->extstress = Rspin * system->extstress * Rspin.transpose();//将系统的外部应力张量 extstress 通过旋转矩阵 Rspin 进行旋转，以将其从当前位错配置的坐标系旋转回全局坐标系，确保后续的应力控制计算使用正确的应力状态
    
    }
    
    if (ctrl.loading == STRAIN_RATE_CONTROL) {
        Mat33 A = outer(edir, edir);//计算应变率控制下的应变增量张量 A，使用应变方向 edir 的外积来构建一个张量，该张量表示在应变率控制下的应变增量的方向和分布
        double dpstrain = dot(system->dEp, A);//计算塑性应变增量 dpstrain，通过将系统的塑性应变增量 dEp 与应变增量张量 A 进行点积，得到在应变率控制下的塑性应变增量的标量值，表示在当前加载条件下由于位错运动引起的塑性变形的增量
        double dstrain = ctrl.erate * system->realdt;//计算总应变增量 dstrain，通过将控制参数中的应变率 erate 与系统的真实时间步长 realdt 相乘，得到在当前时间步长内的总应变增量，表示在当前加载条件下的总变形增量
        double Eyoung = 2.0 * system->params.MU * (1.0 + system->params.NU);//计算杨氏模量 Eyoung，使用系统参数中的剪切模量 MU 和泊松比 NU，通过公式 E = 2 * MU * (1 + NU) 来计算杨氏模量，表示材料的弹性刚度，用于将应变增量转换为应力增量
        double dstress = Eyoung * (dstrain - dpstrain);//计算应力增量 dstress，通过将总应变增量 dstrain 减去塑性应变增量 dpstrain 得到弹性应变增量，然后乘以杨氏模量 Eyoung 来得到对应的应力增量，表示在当前加载条件下的弹性响应部分的应力增量
        
        system->extstress += dstress * A;//将计算得到的应力增量 dstress 乘以应变增量张量 A，得到一个应力增量张量，然后将其加到系统的外部应力张量 extstress 上，更新系统的外部应力状态，以反映在当前加载条件下的应力变化
        Etot += dstrain * A;//将总应变增量 dstrain 乘以应变增量张量 A，得到一个应变增量张量，然后将其加到总应变张量 Etot 上，更新总应变状态，以反映在当前加载条件下的应变变化
        strain = dot(Etot, A);//通过将总应变张量 Etot 与应变增量张量 A 进行点积，计算当前的应变值 strain，表示在当前加载条件下的总应变状态
        stress = dot(system->extstress, A);//通过将系统的外部应力张量 extstress 与应变增量张量 A 进行点积，计算当前的应力值 stress，表示在当前加载条件下的总应力状态
        pstrain += dpstrain;//将计算得到的塑性应变增量 dpstrain 累加到总塑性应变 pstrain 上，更新总塑性应变状态，以反映在当前加载条件下由于位错运动引起的塑性变形的累积增量
    }
    
    if (ctrl.loading == STRESS_CONTROL) {
        strain = 0.0;//在应力控制下，当前的应变值 strain 被设置为 0.0，因为在这种加载方式下，系统的应变状态是由外部应力直接控制的，而不是通过应变率控制计算得到的，因此将其初始化为 0.0，以便后续根据外部应力状态进行更新
        stress = von_mises(system->extstress);//在应力控制下，当前的应力值 stress 通过计算系统的外部应力张量 extstress 的冯·米塞斯等效应力来得到，使用 von_mises 函数将外部应力张量转换为一个标量值，表示在当前加载条件下的等效应力状态，以便后续根据该应力状态进行更新
        double dpstrain = von_mises(system->dEp);//在应力控制下，计算塑性应变增量 dpstrain 通过将系统的塑性应变增量 dEp 的冯·米塞斯等效应力来得到，使用 von_mises 函数将塑性应变增量张量转换为一个标量值，表示在当前加载条件下由于位错运动引起的等效塑性变形的增量，以便后续根据该增量更新总塑性应变状态
        pstrain += dpstrain;//将计算得到的塑性应变增量 dpstrain 累加到总塑性应变 pstrain 上，更新总塑性应变状态，以反映在当前加载条件下由于位错运动引起的塑性变形的累积增量
    }
    
    tottime += system->realdt;//将系统的真实时间步长 realdt 累加到总时间 tottime 上，更新总时间状态，以反映在当前时间步长内的时间进展
    
    if (system->oprec)
        system->oprec->add_op(OpRec::UpdateOutput());//如果启用了操作记录（OpRec），则添加一个更新输出的操作记录，以便在未来的模拟中能够回放或分析该操作记录，确保模拟的可追溯性和可分析性
}//更新力学状态的函数，根据当前的加载控制方式（应变率控制或应力控制）进行相应的计算和更新，包括旋转应变方向和外部应力、计算应变增量和应力增量、更新总应变和总塑性应变、以及更新总时间等，以确保模拟的力学状态与当前加载条件一致，并且在启用操作记录时添加相应的操作记录以便未来分析

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::output()
 *                  Output stress, strain, config, etc.
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::output(Control& ctrl)//输出当前的应力、应变、配置等信息，根据控制参数中的输出频率和输出字段进行相应的输出操作，包括在控制台打印信息、写入属性数据文件、输出配置文件和重启文件，以及记录计时器信息等，以便用户能够监控模拟的进展和分析模拟结果
{
    // Print info in the console
    if (istep%ctrl.printfreq == 0) {
        if (ctrl.loading == STRAIN_RATE_CONTROL) {
            ExaDiS_log("Step = %6d: nodes = %d, dt = %e, strain = %e, elapsed = %.1f sec\n",
            istep, system->Nnodes_total(), system->realdt, strain, timer.seconds());
        } else {
            ExaDiS_log("Step = %6d: nodes = %d, dt = %e, time = %e, elapsed = %.1f sec\n",
            istep, system->Nnodes_total(), system->realdt, tottime, timer.seconds());
        }//根据控制参数中的打印频率 printfreq，在控制台输出当前的步数、节点数量、时间步长、应变或时间，以及模拟已经运行的总时间，以便用户能够监控模拟的进展和性能
    }
    
    // Output properties
    if (istep%ctrl.propfreq == 0 || istep == 1) {
        std::string filename = outputdir+"/stress_strain_dens.dat";
        FILE *fp = fopen(filename.c_str(), "a");//根据控制参数中的属性输出频率 propfreq，在指定的输出目录下以追加模式打开一个名为 "stress_strain_dens.dat" 的文件，如果无法打开则报错退出，以便将当前的应力、应变、密度等属性数据写入该文件中，供后续分析使用
        if (fp == NULL)//如果文件打开失败，输出错误信息并终止程序，以确保后续的属性数据写入操作能够成功进行
            ExaDiS_fatal("Error: cannot open output file %s\n", filename.c_str());//文件打开失败，输出错误信息并终止程序
        bool header = (istep == 0);//如果当前步数是 0，则需要写入文件头，表示属性数据的列名，以便后续分析时能够正确识别每列数据的含义
        if (header) fprintf(fp, "#");//如果需要写入文件头，则在文件开头写入一个注释符号 "#"，以便后续分析时能够识别该行是文件头而不是数据行
        for (auto field : ctrl.props) {
            if (field == Prop::STEP) if (header) { fprintf(fp, " Step"); } else { fprintf(fp, "%d ", istep); }
            else if (field == Prop::STRAIN) if (header) { fprintf(fp, " Strain"); } else { fprintf(fp, "%e ", strain); }
            else if (field == Prop::STRESS) if (header) { fprintf(fp, " Stress"); } else { fprintf(fp, "%e ", stress); }
            else if (field == Prop::DENSITY) if (header) { fprintf(fp, " Density"); } else { fprintf(fp, "%e ", system->density); }
            else if (field == Prop::NNODES) if (header) { fprintf(fp, " Nnodes"); } else { fprintf(fp, "%d ", system->Nnodes_total()); }
            else if (field == Prop::NSEGS) if (header) { fprintf(fp, " Nsegs"); } else { fprintf(fp, "%d ", system->Nsegs_total()); }
            else if (field == Prop::DT) if (header) { fprintf(fp, " dt"); } else { fprintf(fp, "%e ", system->realdt); }
            else if (field == Prop::TIME) if (header) { fprintf(fp, " Time"); } else { fprintf(fp, "%e ", tottime); }
            else if (field == Prop::WALLTIME) if (header) { fprintf(fp, " Walltime"); } else { fprintf(fp, "%e ", timer.seconds()); }
            else if (field == Prop::EDIR) if (header) { fprintf(fp, " edirx ediry edirz"); } else { fprintf(fp, "%e %e %e ", edir.x, edir.y, edir.z); }//根据控制参数中的属性列表 props，循环检查每个属性字段，并根据是否需要写入文件头来决定写入列名还是写入对应的属性值，以便在 "stress_strain_dens.dat" 文件中记录当前的应力、应变、密度等属性数据，供后续分析使用
            else if (field == Prop::RORIENT) {
                Mat33 R = system->crystal.R;
                if (header) fprintf(fp, " Rxx Rxy Rxz Ryx Ryy Ryz Rzx Rzy Rzz"); 
                else fprintf(fp, "%e %e %e %e %e %e %e %e %e ", R.xx(), R.xy(), R.xz(), R.yx(), R.yy(), R.yz(), R.zx(), R.zy(), R.zz());//如果属性字段是晶体旋转矩阵 RORIENT，则根据是否需要写入文件头来决定写入列名还是写入当前晶体旋转矩阵的9个分量，以便在 "stress_strain_dens.dat" 文件中记录当前的晶体旋转状态，供后续分析使用
            } else if (field == Prop::ALLSTRESS) {
                if (header) fprintf(fp, " Sxx Syy Szz Sxy Sxz Syz"); 
                else fprintf(fp, "%e %e %e %e %e %e ", system->extstress.xx(), system->extstress.yy(), system->extstress.zz(), 
                                                       system->extstress.xy(), system->extstress.xz(), system->extstress.yz());
            }//如果属性字段是 ALLSTRESS，则根据是否需要写入文件头来决定写入列名还是写入当前外部应力张量的6个独立分量，以便在 "stress_strain_dens.dat" 文件中记录当前的外部应力状态，供后续分析使用
        }
        fprintf(fp, "\n");//在文件中写入一个换行符，表示当前步的属性数据已经写入完成，以便后续分析时能够正确识别每行数据的边界
        fclose(fp);//关闭属性数据文件，完成写入操作，以确保数据被正确保存到文件中，供后续分析使用
    }
    
    // Output configuration
    bool out = (ctrl.outfreqdt > 0.0) ? (tottime >= outfiletime+ctrl.outfreqdt) : (istep%ctrl.outfreq == 0);
    if (out || istep == 0) {
        int outfilenum = istep;
        if (ctrl.outfreqdt > 0.0) {
            outfiletime = tottime;
            outfilenum = outfilecounter++;
        }//根据控制参数中的输出频率 outfreq 和输出时间频率 outfreqdt，判断是否需要输出当前的配置文件，如果需要则计算输出文件的编号，并更新下一个输出时间，以便在未来的模拟中能够按照指定的频率继续输出配置文件
        
        system->write_config(outputdir+"/config."+std::to_string(outfilenum)+".data");//调用系统的 write_config 函数，将当前的配置状态写入一个名为 "config.<outfilenum>.data" 的文件中，其中 <outfilenum> 是根据当前步数或输出时间计算得到的输出文件编号，以便在未来的模拟中能够按照指定的频率继续输出配置文件
        
        // Restart files
        write_restart(outputdir+"/restart."+std::to_string(outfilenum)+".exadis");//调用 write_restart 函数，将当前的模拟状态写入一个名为 "restart.<outfilenum>.exadis" 的重启文件中，其中 <outfilenum> 是根据当前步数或输出时间计算得到的输出文件编号，以便在未来的模拟中能够从该重启文件继续模拟
        
        // Timers
        if (istep > 0) {
            std::string filename;
            if (timeronefile) filename = outputdir+"/timer.dat";
            else filename = outputdir+"/timer."+std::to_string(outfilenum)+".dat";
            FILE *fp = fopen(filename.c_str(), "a");//根据是否将计时器信息写入一个文件 timer.dat 来决定输出文件的名称，如果无法打开则报错退出，以便将当前的计时器信息写入该文件中，供后续分析使用
            if (fp == NULL)
                ExaDiS_fatal("Error: cannot open output file %s\n", filename.c_str());
            fprintf(fp, "Step = %d\n", istep);//在计时器文件中写入当前的步数，以便后续分析时能够识别每段计时器信息对应的模拟步数
            fprintf(fp, "\n");//在计时器文件中写入一个空行，以便后续分析时能够更清晰地分隔不同步数的计时器信息
            fprintf(fp, "Force time:        %12.3f sec\n", system->timer[system->TIMER_FORCE].accumtime);//在计时器文件中写入力计算的累计时间，以便后续分析时能够评估力计算的性能
            fprintf(fp, "Mobility time:     %12.3f sec\n", system->timer[system->TIMER_MOBILITY].accumtime);
            fprintf(fp, "Integration time:  %12.3f sec\n", system->timer[system->TIMER_INTEGRATION].accumtime);
            fprintf(fp, "Cross-slip time:   %12.3f sec\n", system->timer[system->TIMER_CROSSSLIP].accumtime);
            fprintf(fp, "Collision time:    %12.3f sec\n", system->timer[system->TIMER_COLLISION].accumtime);
            fprintf(fp, "Topology time:     %12.3f sec\n", system->timer[system->TIMER_TOPOLOGY].accumtime);
            fprintf(fp, "Remesh time:       %12.3f sec\n", system->timer[system->TIMER_REMESH].accumtime);//在计时器文件中写入各个模块的累计时间，包括力计算、移动性计算、时间积分、交滑移、碰撞检测、拓扑处理和重网格化等，以便后续分析时能够评估各个模块的性能
            fprintf(fp, "Output time:       %12.3f sec\n", system->timer[system->TIMER_OUTPUT].accumtime);
            fprintf(fp, "\n");
            fprintf(fp, "Total time:        %12.3f sec\n", timer.seconds());//在计时器文件中写入总的模拟时间，以便后续分析时能够评估整个模拟的性能
            if (system->numdevtimer > 0) {
                fprintf(fp, "\n\n");
                fprintf(fp, "Additional (development) timers\n");
                fprintf(fp, "\n");
                for (int i = 0; i < system->numdevtimer; i++)
                    fprintf(fp, "%s time: %.3f sec\n", system->devtimer[i].label.c_str(), system->devtimer[i].accumtime);
            }//如果系统中定义了额外的开发计时器，则在计时器文件中写入这些计时器的信息，包括计时器的标签和累计时间，以便后续分析时能够评估这些额外计时器的性能
            if (timeronefile)//如果将计时器信息写入一个文件 timer.dat，则在该文件中写入一个分隔线，以便后续分析时能够更清晰地分隔不同步数的计时器信息
                fprintf(fp, "\n--------------------------------------------------\n");
            fclose(fp);
        }
    }
    
    // OpRec output
    if (system->oprec && ctrl.oprecwritefreq > 0 && istep > 0) {
        if (istep % ctrl.oprecwritefreq == 0)
            system->oprec->write_file(outputdir+"/oprec."+std::to_string(system->oprec->filecounter)+".exadis");//根据控制参数中的操作记录输出频率 oprecwritefreq，在指定的输出目录下以 ".exadis" 格式写入一个名为 "oprec.<filecounter>.exadis" 的操作记录文件，其中 <filecounter> 是系统的操作记录文件计数器，以便在未来的模拟中能够按照指定的频率继续输出操作记录文件
        
        if (istep % ctrl.oprecfilefreq == 0)
            system->oprec->filecounter++;//根据控制参数中的操作记录文件频率 oprecfilefreq，在系统的操作记录文件计数器 filecounter 上进行递增，以便在未来的模拟中能够按照指定的频率继续输出操作记录文件，并且确保每个操作记录文件都有一个唯一的编号
    }//如果启用了操作记录（OpRec）并且控制参数中的操作记录输出频率 oprecwritefreq 大于 0，并且当前步数大于 0，则根据控制参数中的操作记录文件频率 oprecfilefreq 来决定是否递增操作记录文件计数器 filecounter，以便在未来的模拟中能够按照指定的频率继续输出操作记录文件，并且确保每个操作记录文件都有一个唯一的编号
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::Stepper::iterate()
 *                  Controls the number of steps in a run
 *
 *-------------------------------------------------------------------------*/
bool ExaDiSApp::Stepper::iterate(ExaDiSApp* exadis)//根据指定的停止条件类型（步数、应变、时间或壁钟时间）来控制模拟的迭代过程，判断是否继续迭代，并在第一次调用时进行相应的初始化，以确保模拟能够按照用户指定的条件正确地进行迭代
{    
    if (exadis->init) {
        if (type == NUM_STEPS) maxsteps += exadis->istep;//如果停止条件类型是 NUM_STEPS，则将当前的步数 istep 加到最大步数 maxsteps 上，以便在未来的迭代过程中能够正确地判断是否达到指定的总步数
        if (type == NUM_STEPS || type == MAX_STEPS) ExaDiS_log("Run for %d steps\n", maxsteps - exadis->istep);//如果停止条件类型是 NUM_STEPS 或 MAX_STEPS，则在日志中输出还需要运行的步数，以便用户能够了解模拟的进展和剩余的迭代次数
        if (type == MAX_STRAIN) ExaDiS_log("Run until reaching strain = %f\n", stopval);//如果停止条件类型是 MAX_STRAIN，则在日志中输出还需要达到的应变值，以便用户能够了解模拟的进展和剩余的应变增量
        if (type == MAX_TIME) ExaDiS_log("Run until reaching time = %f\n", stopval);//如果停止条件类型是 MAX_TIME，则在日志中输出还需要达到的时间值，以便用户能够了解模拟的进展和剩余的时间增量
        if (type == MAX_WALLTIME) ExaDiS_log("Run until reaching wall time = %f sec\n", stopval);//如果停止条件类型是 MAX_WALLTIME，则在日志中输出还需要达到的壁钟时间值，以便用户能够了解模拟的进展和剩余的时间增量
        exadis->init = false;//将初始化标志 init 设置为 false，以便在后续的迭代过程中不再进行上述的初始化操作，确保模拟能够按照用户指定的条件正确地进行迭代
    }
    
    bool iterate = false;
    if (exadis->system->Nnodes_total() == 0 || exadis->system->Nsegs_total() == 0) {
        ExaDiS_log("No dislocation in the system. Stopping.\n");
        iterate = false;
    }//如果系统中没有位错节点或线段，则在日志中输出提示信息 "No dislocation in the system. Stopping."，并将迭代标志 iterate 设置为 false，以便停止模拟的迭代过程，因为没有位错存在意味着没有物理过程需要模拟
    else if (type == NUM_STEPS || type == MAX_STEPS) iterate = (exadis->istep < maxsteps);//如果停止条件类型是 NUM_STEPS 或 MAX_STEPS，则判断当前的步数 istep 是否小于最大步数 maxsteps，如果是则将迭代标志 iterate 设置为 true，以便继续模拟的迭代过程，否则将其设置为 false，以便停止模拟的迭代过程
    else if (type == MAX_STRAIN) iterate = (fabs(exadis->strain) < stopval);//如果停止条件类型是 MAX_STRAIN，则判断当前的应变值 strain 的绝对值是否小于停止值 stopval，如果是则将迭代标志 iterate 设置为 true，以便继续模拟的迭代过程，否则将其设置为 false，以便停止模拟的迭代过程
    else if (type == MAX_TIME) iterate = (exadis->tottime < stopval);//如果停止条件类型是 MAX_TIME，则判断当前的总时间 tottime 是否小于停止值 stopval，如果是则将迭代标志 iterate 设置为 true，以便继续模拟的迭代过程，否则将其设置为 false，以便停止模拟的迭代过程
    else if (type == MAX_WALLTIME) iterate = (exadis->timer.seconds() < stopval);//如果停止条件类型是 MAX_WALLTIME，则判断当前的模拟时间（通过计时器 timer 的 seconds() 方法获取）是否小于停止值 stopval，如果是则将迭代标志 iterate 设置为 true，以便继续模拟的迭代过程，否则将其设置为 false，以便停止模拟的迭代过程
    if (iterate) exadis->istep++;//如果迭代标志 iterate 为 true，则将当前的步数 istep 进行递增，以便在下一次迭代时能够正确地反映当前的模拟步数，确保模拟能够按照用户指定的条件正确地进行迭代
    return iterate;//返回迭代标志 iterate，指示是否继续模拟的迭代过程，以便在主循环中根据该标志来控制模拟的运行
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::initialize()
 *                  Iniatialize and check that everything is setup properly
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::initialize(Control& ctrl, bool check_modules)//初始化函数，负责检查系统和模块的设置是否正确，并进行必要的初始化操作，以确保模拟能够顺利进行，包括检查系统对象是否定义、检查各个模块是否定义（如果 check_modules 标志为 true）、检查系统参数的有效性、设置模拟状态、处理重启文件、激活操作记录（OpRec）等，以便在后续的模拟过程中能够正确地执行各个模块的功能，并且能够按照用户指定的条件进行模拟
{
    // Required modules
    if (system == nullptr) ExaDiS_fatal("Error: undefined system\n");//检查系统对象是否定义，如果未定义则输出错误信息并终止程序，以确保后续的模拟操作能够在一个有效的系统对象上进行
    if (check_modules) {
        if (force == nullptr) ExaDiS_fatal("Error: undefined force module\n");
        if (mobility == nullptr) ExaDiS_fatal("Error: undefined mobility module\n");
        if (integrator == nullptr) ExaDiS_fatal("Error: undefined integrator module\n");
        if (collision == nullptr) ExaDiS_fatal("Error: undefined collision module\n");
        if (topology == nullptr) ExaDiS_fatal("Error: undefined topology module\n");
        if (remesh == nullptr) ExaDiS_fatal("Error: undefined remesh module\n");
    }//如果 check_modules 标志为 true，则检查各个模块（力模块、移动性模块、积分器模块、碰撞模块、拓扑模块和重网格化模块）是否定义，如果未定义则输出错误信息并终止程序，以确保后续的模拟操作能够在一个完整的模块集合上进行
    
    system->params.check_params();//检查系统参数的有效性，调用系统参数对象的 check_params 方法来验证参数的设置是否合理，如果参数设置不正确则可能会导致模拟过程中出现错误或不稳定，因此进行参数检查是确保模拟能够顺利进行的重要步骤
    if (!setup) set_simulation();//如果模拟设置尚未完成，则调用 set_simulation 函数来进行必要的模拟设置，以确保模拟能够按照用户指定的条件进行，包括设置输出目录、初始化随机数生成器、设置初始应变方向等，以便在后续的模拟过程中能够正确地执行各个模块的功能，并且能够按照用户指定的条件进行模拟
    
    if (!restart) {
        system->extstress = ctrl.appstress;
        edir = ctrl.edir.normalized();
    }//如果不是从重启文件开始模拟，则将系统的外部应力设置为控制参数中的应用应力 appstress，并将应变方向 edir 设置为控制参数中的应变方向 edir 的归一化版本，以确保在非重启情况下模拟能够按照用户指定的加载条件进行
    
    // OpRec
    if (ctrl.oprecwritefreq > 0) {
        if (ctrl.oprecfilefreq <= 0)
            ctrl.oprecfilefreq = ctrl.oprecwritefreq;
        system->oprec->activate();
    }//如果控制参数中的操作记录输出频率 oprecwritefreq 大于 0，则检查操作记录文件频率 oprecfilefreq 是否设置，如果未设置则将其设置为与 oprecwritefreq 相同，并激活系统的操作记录功能，以便在后续的模拟过程中能够按照指定的频率输出操作记录文件，确保模拟的可追溯性和可分析性
    
    system->oprec->add_op(OpRec::Initialize("Serial", Vec3i(1)));//添加一个初始化操作记录，记录模拟的初始化过程，包括使用的并行模式（Serial）和处理器网格大小（Vec3i(1)），以便在未来的模拟中能够回放或分析该操作记录，确保模拟的可追溯性和可分析性
    
    init = true;//将初始化标志 init 设置为 true，以便在后续的模拟过程中能够正确地进行第一次迭代的初始化操作，确保模拟能够按照用户指定的条件正确地进行迭代
    restart = false;//将重启标志 restart 设置为 false，以便在后续的模拟过程中不再进行重启文件的读取操作，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Initial output at step 0
    output(ctrl);//在初始化完成后，调用 output 函数进行一次初始输出，记录当前的应力、应变、配置等信息，以便在后续的模拟过程中能够监控模拟的进展和分析模拟结果，并且确保在第一次迭代之前就有一个完整的输出记录
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::step()
 *                  Genereric DDD single simulation step
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::step(Control& ctrl)//执行一个通用的离散位错动力学（DDD）模拟步骤，按照以下顺序进行：首先进行力的预计算，然后计算节点上的力，接着计算移动性，进行时间积分，计算塑性应变，重置滑移面，处理交滑移、碰撞和拓扑变化，进行重网格化，更新力学状态，并最后进行输出，以确保在每个模拟步骤中能够正确地执行各个模块的功能，并且能够按照用户指定的条件进行模拟
{
    // Do some force pre-computation for the step if needed
    force->pre_compute(system);//调用力模块的 pre_compute 方法，对系统进行一些力的预计算，以便在后续的节点力计算中能够更高效地获取所需的力信息，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Nodal force calculation
    force->compute(system);//调用力模块的 compute 方法，计算系统中每个节点上的力，包括位错线段之间的相互作用力、外部应力引起的力等，以便在后续的移动性计算和时间积分中能够使用这些力信息来更新节点的位置和状态，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Mobility calculation
    system->pstrain = pstrain; // Junjie: update plastic strain in system
    mobility->compute(system);//调用移动性模块的 compute 方法，计算系统中每个节点的移动性，根据节点上的力和系统的物理参数来确定节点的速度，以便在后续的时间积分中能够使用这些速度信息来更新节点的位置和状态，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Time-integration
    integrator->integrate(system);//调用积分器模块的 integrate 方法，进行时间积分，根据节点的速度和系统的时间步长来更新节点的位置和状态，以便在后续的模拟过程中能够正确地反映节点的运动和系统的演化，确保模拟能够按照用户指定的条件正确地进行迭代
    oprec_save_integration(ctrl);//调用 oprec_save_integration 函数，保存当前的积分操作记录，以便在未来的模拟中能够回放或分析该操作记录，确保模拟的可追溯性和可分析性
    
    // Compute plastic strain
    system->plastic_strain();//调用系统的 plastic_strain 方法，计算系统中的塑性应变，根据节点的运动和位错的演化来更新系统的塑性应变状态，以便在后续的模拟过程中能够正确地反映由于位错运动引起的塑性变形，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Reset glide planes
    system->reset_glide_planes();//调用系统的 reset_glide_planes 方法，重置系统中每个节点的滑移面，根据当前的位错配置和节点的位置来更新每个节点的滑移面信息，以便在后续的模拟过程中能够正确地反映节点的滑移行为和位错的演化，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Cross-slip
    if (crossslip)
        crossslip->handle(system);//如果交滑移模块 crossslip 已经定义，则调用其 handle 方法，处理系统中的交滑移现象，根据当前的位错配置和节点的信息来判断是否发生交滑移，并进行相应的处理，以便在后续的模拟过程中能够正确地反映交滑移对位错演化的影响，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Collision
    collision->handle(system);//调用碰撞模块的 handle 方法，处理系统中的碰撞现象，根据当前的位错配置和节点的信息来判断是否发生碰撞，并进行相应的处理，以便在后续的模拟过程中能够正确地反映碰撞对位错演化的影响，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Topology
    topology->handle(system);//调用拓扑模块的 handle 方法，处理系统中的拓扑变化，根据当前的位错配置和节点的信息来判断是否发生拓扑变化，并进行相应的处理，以便在后续的模拟过程中能够正确地反映拓扑变化对位错演化的影响，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Remesh
    remesh->remesh(system);//调用重网格化模块的 remesh 方法，对系统进行重网格化，根据当前的位错配置和节点的信息来判断是否需要进行重网格化，并进行相应的处理，以便在后续的模拟过程中能够正确地反映重网格化对位错演化的影响，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Update stress
    update_mechanics(ctrl);//调用 update_mechanics 函数，更新系统的力学状态，根据当前的加载条件和位错的演化来计算和更新系统的应力、应变、塑性应变等力学属性，以便在后续的模拟过程中能够正确地反映当前的力学状态，确保模拟能够按照用户指定的条件正确地进行迭代
    
    // Output
    output(ctrl);//调用 output 函数，输出当前的应力、应变、配置等信息，根据控制参数中的输出频率和输出字段进行相应的输出操作，以便在后续的模拟过程中能够监控模拟的进展和分析模拟结果，并且确保在每个模拟步骤中都有一个完整的输出记录
}

/*---------------------------------------------------------------------------
 *
 *    Function:     ExaDiSApp::run()
 *                  Generic DDD cycle to run a simulation for a number
 *                  of steps under the conditions prescribed in the 
 *                  Control argument.
 *
 *-------------------------------------------------------------------------*/
void ExaDiSApp::run(Control& ctrl)//运行一个通用的离散位错动力学（DDD）模拟循环，根据控制参数中的条件来执行模拟的迭代过程，包括初始化、主循环和计时器信息输出等，以确保模拟能够按照用户指定的条件正确地进行，并且能够监控模拟的进展和性能
{
    // Iniatialize and check that everything is setup properly
    initialize(ctrl);//调用 initialize 函数，进行模拟的初始化和检查，确保系统和模块的设置正确，并且能够按照用户指定的条件进行模拟，以便在后续的模拟过程中能够正确地执行各个模块的功能，并且能够按照用户指定的条件进行模拟
    
    // Main loop
    timer.reset();
    while (ctrl.nsteps.iterate(this)) {
        step(ctrl);
    }//在主循环中，调用控制参数中的停止条件的 iterate 方法来判断是否继续迭代，如果继续则执行一个模拟步骤 step(ctrl)，直到达到指定的停止条件为止，以确保模拟能够按照用户指定的条件正确地进行迭代
    
    Kokkos::fence();//在模拟完成后，调用 Kokkos 的 fence 函数来确保所有的并行操作都已经完成，以便在输出计时器信息之前能够正确地获取到所有的计时器数据，确保模拟的性能评估是准确的
    double totaltime = timer.seconds();//获取总的模拟时间，通过计时器 timer 的 seconds() 方法来获取，以便在后续的日志输出中能够显示整个模拟的运行时间，供用户评估模拟的性能
    system->print_timers(totaltime);//调用系统的 print_timers 方法，打印各个模块的计时器信息，包括力计算、移动性计算、时间积分、交滑移、碰撞检测、拓扑处理和重网格化等，以及总的模拟时间，以便用户能够评估各个模块的性能以及整个模拟的性能
    ExaDiS_log("RUN TIME: %f sec\n", totaltime);//在日志中输出整个模拟的运行时间，以便用户能够评估模拟的性能，并且为后续的模拟提供一个性能基准，供用户参考和比较
}

} // namespace ExaDiS

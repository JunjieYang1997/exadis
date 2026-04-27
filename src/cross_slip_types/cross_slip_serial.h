/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  This module implements cross-slip for FCC crystals in serial fashion.
 *  It is a direct translation in ExaDiS of ParaDiS source file
 *  ParaDiS/src/CrossSlipFCC.cc.
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CROSS_SLIP_SERIAL_H
#define EXADIS_CROSS_SLIP_SERIAL_H

#include "force.h"
#include "cross_slip.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:        CrossSlipSerial
 *
 *-------------------------------------------------------------------------*/
class CrossSlipSerial : public CrossSlip {
private:
    Force* force;
    
public:
    CrossSlipSerial(System* system, Force* _force) : force(_force) {}
    
    void handle(System* system);
    
    const char* name() { return "CrossSlipSerial"; }
};//定义一个类CrossSlipSerial，继承自CrossSlip，包含一个指向Force类型的指针作为成员变量，并实现了构造函数和handle函数，以及一个返回类名的name函数,这就是一个表头文件，具体的实现细节在对应的源文件cross_slip_serial.cpp中

} // namespace ExaDiS

#endif

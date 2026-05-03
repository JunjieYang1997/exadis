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
    // BCC 热激活交滑移参数（无非施密特修正）
    // 温度随塑性应变升高：T = kT * pstrain + bT，与 MobilityBCC0b_temp 完全一致
    struct BCCCrossSlipParams {
        double kT          = 0.0;    // 升温斜率 [K/strain]，与 mobility 的 kT 相同
        double bT          = 300.0;  // 初始温度 [K]，与 mobility 的 bT 相同
        double delta_H_cs  = 1.63;   // 零应力激活焓 [eV]
        double tau_P_cs    = 960e6;  // 交滑移面 Peierls 应力 [Pa]
        double p_shape     = 0.86;   // Peierls 势形状参数 p
        double q_shape     = 1.69;   // Peierls 势形状参数 q
        double delta_S_cs  = 0.0;    // 激活熵 [eV/K]
        double omega_D     = 1e13;   // Debye 频率 [s^-1]
        double eps_dot_sim = 1e4;    // 模拟应变率 [s^-1]
        double eps_dot_exp = 1e-3;   // 实验应变率 [s^-1]
        double L0_ref      = 1e-6;   // 参考位错长度 [m]
        double tau_f_cs    = 50e6;   // 交滑移面摩擦应力 [Pa]
    };
    BCCCrossSlipParams bcc_params;

    CrossSlipSerial(System* system, Force* _force) : force(_force) {}
    CrossSlipSerial(System* system, Force* _force, BCCCrossSlipParams _bcc_params)
        : force(_force), bcc_params(_bcc_params) {}

    void handle(System* system);

    const char* name() { return "CrossSlipSerial"; }
};

} // namespace ExaDiS

#endif

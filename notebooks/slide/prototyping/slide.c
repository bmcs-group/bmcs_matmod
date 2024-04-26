#include <windows.h>
#include <float.h>
#include <math.h>
#include <excpt.h>
#include <stdio.h>
//#include <gsl/gsl_math.h>
//#include <gsl/gsl_eigen.h>



#define UserMaterialParamsCount 17
/** All user material parameter names in an array, which is used
 *  in \ref UserDLLMaterialParamName()
 *
 *  Not a compulsory part of the interface - the user may implement
 *  the name returning function in another way.
 *
 */

char* UserMaterialParamNames[UserMaterialParamsCount] =
{
	"E_N",
	"E_T",
	"gamma_N",
	"gamma_T",
	"K_T",
	"S_N",
	"S_T",
	"f_c0",
	"f_c",
	"f_s",
	"f_t",
	"c_N",
	"r_N",
	"c_T",
	"r_T",
	"m", 
	"eta"
};
/** Number of user state variables = floating point
 *  values stored in each material point.
 *  This value is also returned by \ref UserDLLStateVarsCount()
 *
 *  Not a compulsory part of the interface - the user may implement
 *  the name returning function in another way.
 */

void get_Sig(double u_Ty_, double E_N_, double E_T_, double K_T_, double alpha_Tx_, double alpha_Ty_, double gamma_T_, double omega_N_, double omega_T_, double sigma_p_N_, double u_p_N_, double u_p_Tx_, double u_p_Ty_, double u_N_, double u_Tx_, double z_T_, double* out_3425652542372553079) {


    out_3425652542372553079[0] = -1.0 / 2.0 * E_T_ * (1 - omega_T_) * (2 * u_p_Tx_ - 2 * u_Tx_);
    out_3425652542372553079[1] = -1.0 / 2.0 * E_T_ * (1 - omega_T_) * (-2 * u_Ty_ + 2 * u_p_Ty_);
    out_3425652542372553079[2] = -1.0 / 2.0 * E_N_ * (2 * u_p_N_ - 2 * u_N_) * (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1);
    out_3425652542372553079[3] = K_T_ * z_T_;
    out_3425652542372553079[4] = alpha_Tx_ * gamma_T_;
    out_3425652542372553079[5] = alpha_Ty_ * gamma_T_;
    out_3425652542372553079[6] = (1.0 / 2.0) * E_T_ * pow(u_Ty_ - u_p_Ty_, 2) + (1.0 / 2.0) * E_T_ * pow(-u_p_Tx_ + u_Tx_, 2);
    out_3425652542372553079[7] = (1.0 / 2.0) * E_N_ * pow(-u_p_N_ + u_N_, 2) * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                )));

}


void get_dSig_dEps(double u_Ty_, double E_N_, double E_T_, double K_T_, double gamma_T_, double omega_N_, double omega_T_, double sigma_p_N_, double u_p_N_, double u_p_Tx_, double u_p_Ty_, double u_N_, double u_Tx_, double* out_1055060450990087600) {

    out_1055060450990087600[0] = -E_T_ * (1 - omega_T_);
    out_1055060450990087600[1] = 0;
    out_1055060450990087600[2] = 0;
    out_1055060450990087600[3] = 0;
    out_1055060450990087600[4] = 0;
    out_1055060450990087600[5] = 0;
    out_1055060450990087600[6] = (1.0 / 2.0) * E_T_ * (2 * u_p_Tx_ - 2 * u_Tx_);
    out_1055060450990087600[7] = 0;
    out_1055060450990087600[8] = 0;
    out_1055060450990087600[9] = -E_T_ * (1 - omega_T_);
    out_1055060450990087600[10] = 0;
    out_1055060450990087600[11] = 0;
    out_1055060450990087600[12] = 0;
    out_1055060450990087600[13] = 0;
    out_1055060450990087600[14] = (1.0 / 2.0) * E_T_ * (-2 * u_Ty_ + 2 * u_p_Ty_);
    out_1055060450990087600[15] = 0;
    out_1055060450990087600[16] = 0;
    out_1055060450990087600[17] = 0;
    out_1055060450990087600[18] = -E_N_ * (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1);
    out_1055060450990087600[19] = 0;
    out_1055060450990087600[20] = 0;
    out_1055060450990087600[21] = 0;
    out_1055060450990087600[22] = 0;
    out_1055060450990087600[23] = (1.0 / 2.0) * E_N_ * (2 * u_p_N_ - 2 * u_N_) * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                )));
    out_1055060450990087600[24] = 0;
    out_1055060450990087600[25] = 0;
    out_1055060450990087600[26] = 0;
    out_1055060450990087600[27] = K_T_;
    out_1055060450990087600[28] = 0;
    out_1055060450990087600[29] = 0;
    out_1055060450990087600[30] = 0;
    out_1055060450990087600[31] = 0;
    out_1055060450990087600[32] = 0;
    out_1055060450990087600[33] = 0;
    out_1055060450990087600[34] = 0;
    out_1055060450990087600[35] = 0;
    out_1055060450990087600[36] = gamma_T_;
    out_1055060450990087600[37] = 0;
    out_1055060450990087600[38] = 0;
    out_1055060450990087600[39] = 0;
    out_1055060450990087600[40] = 0;
    out_1055060450990087600[41] = 0;
    out_1055060450990087600[42] = 0;
    out_1055060450990087600[43] = 0;
    out_1055060450990087600[44] = 0;
    out_1055060450990087600[45] = gamma_T_;
    out_1055060450990087600[46] = 0;
    out_1055060450990087600[47] = 0;
    out_1055060450990087600[48] = (1.0 / 2.0) * E_T_ * (2 * u_p_Tx_ - 2 * u_Tx_);
    out_1055060450990087600[49] = (1.0 / 2.0) * E_T_ * (-2 * u_Ty_ + 2 * u_p_Ty_);
    out_1055060450990087600[50] = 0;
    out_1055060450990087600[51] = 0;
    out_1055060450990087600[52] = 0;
    out_1055060450990087600[53] = 0;
    out_1055060450990087600[54] = 0;
    out_1055060450990087600[55] = 0;
    out_1055060450990087600[56] = 0;
    out_1055060450990087600[57] = 0;
    out_1055060450990087600[58] = (1.0 / 2.0) * E_N_ * (2 * u_p_N_ - 2 * u_N_) * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                )));
    out_1055060450990087600[59] = 0;
    out_1055060450990087600[60] = 0;
    out_1055060450990087600[61] = 0;
    out_1055060450990087600[62] = 0;
    out_1055060450990087600[63] = 0;

}


double get_f(double X_Tx_, double X_Ty_, double Z_T_, double omega_N_, double omega_T_, double sigma_p_Tx_, double sigma_p_Ty_, double sigma_p_N_, double f_c0_, double f_c_, double f_s_, double f_t_, double m_) {

    double get_f_result;
    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        get_f_result = sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) - (pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2)) / (Z_T_ + f_s_ - 2 * f_t_ * m_);
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        get_f_result = sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) - (pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2)) / (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_);
    }
    else {
        get_f_result = -Z_T_ + sigma_p_N_ * m_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - f_s_ + sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2));
    }
    return get_f_result;

}

void get_df_dSig(double X_Tx_, double X_Ty_, double Z_T_, double omega_N_, double omega_T_, double sigma_p_Tx_, double sigma_p_Ty_, double sigma_p_N_, double f_c0_, double f_c_, double f_s_, double f_t_, double m_, double* out_2760537359831697957) {

    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_2760537359831697957[0] = (-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((1 - omega_T_) * (Z_T_ + f_s_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_2760537359831697957[0] = (-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((1 - omega_T_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * (Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_));
    }
    else {
        out_2760537359831697957[0] = (-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_)) / ((1 - omega_T_) * sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)));
    }
    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_2760537359831697957[1] = (-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((1 - omega_T_) * (Z_T_ + f_s_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_2760537359831697957[1] = (-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((1 - omega_T_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * (Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_));
    }
    else {
        out_2760537359831697957[1] = (-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_)) / ((1 - omega_T_) * sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)));
    }
    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_2760537359831697957[2] = (1.0 / 2.0) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (2 * omega_N_ * sigma_p_N_ * DiracDelta(sigma_p_N_) / pow(-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1, 2) + 2 / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                    0
                                    )
                                    : ((sigma_p_N_ == 0) ? (
                                        1.0 / 2.0
                                        )
                                        : (
                                            1
                                            ))) + 1)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                0
                                                )
                                                : ((sigma_p_N_ == 0) ? (
                                                    1.0 / 2.0
                                                    )
                                                    : (
                                                        1
                                                        ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * pow(Z_T_ + f_s_ - f_t_ * m_, 2));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_2760537359831697957[2] = (1.0 / 2.0) * (2 * omega_N_ * sigma_p_N_ * DiracDelta(sigma_p_N_) / pow(-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1, 2) + 2 / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1)) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                    0
                                    )
                                    : ((sigma_p_N_ == 0) ? (
                                        1.0 / 2.0
                                        )
                                        : (
                                            1
                                            ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                0
                                                )
                                                : ((sigma_p_N_ == 0) ? (
                                                    1.0 / 2.0
                                                    )
                                                    : (
                                                        1
                                                        ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * pow(Z_T_ + f_c_ * m_ + f_s_, 2));
    }
    else {
        out_2760537359831697957[2] = omega_N_ * sigma_p_N_ * m_ * DiracDelta(sigma_p_N_) / pow(-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1, 2) + m_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1);
    }
    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_2760537359831697957[3] = -(2 * Z_T_ + 2 * f_s_ - 2 * f_t_ * m_) / (Z_T_ + f_s_ - 2 * f_t_ * m_) + (pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2)) / pow(Z_T_ + f_s_ - 2 * f_t_ * m_, 2) + (-m_ * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(-Z_T_ - f_s_ + 2 * f_t_ * m_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + (1.0 / 2.0) * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * (4 * Z_T_ + 4 * f_s_ - 4 * f_t_ * m_) * (pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2)) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) - (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 3)) - 1.0 / 2.0 * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * pow(Z_T_ + f_s_ - 2 * f_t_ * m_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) - 1.0 / 2.0 * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(Z_T_ + f_s_, 2) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + (1.0 / 2.0) * pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * (4 * Z_T_ + 4 * f_s_ - 4 * f_t_ * m_) * (pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2)) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) - pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                    0
                                    )
                                    : ((sigma_p_N_ == 0) ? (
                                        1.0 / 2.0
                                        )
                                        : (
                                            1
                                            ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 3))) / sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                0
                                                )
                                                : ((sigma_p_N_ == 0) ? (
                                                    1.0 / 2.0
                                                    )
                                                    : (
                                                        1
                                                        ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_2760537359831697957[3] = -(2 * Z_T_ + 2 * f_c_ * m_ + 2 * f_s_) / (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_) + (pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2)) / pow(Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_, 2) + (-m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * pow(-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_, 2)) + (1.0 / 2.0) * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * (4 * Z_T_ + 4 * f_c_ * m_ + 4 * f_s_) * (pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2)) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) - 1.0 / 2.0 * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * pow(Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_, 2)) - (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 3) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) - 1.0 / 2.0 * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(Z_T_ + f_c0_ * m_ + f_s_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + (1.0 / 2.0) * (4 * Z_T_ + 4 * f_c_ * m_ + 4 * f_s_) * pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * (pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2)) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2)) - pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                    0
                                    )
                                    : ((sigma_p_N_ == 0) ? (
                                        1.0 / 2.0
                                        )
                                        : (
                                            1
                                            ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 3))) / sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                0
                                                )
                                                : ((sigma_p_N_ == 0) ? (
                                                    1.0 / 2.0
                                                    )
                                                    : (
                                                        1
                                                        ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2)));
    }
    else {
        out_2760537359831697957[3] = -1;
    }
    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_2760537359831697957[4] = (1.0 / 2.0) * (2 * X_Tx_ - 2 * sigma_p_Tx_ / (1 - omega_T_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_2760537359831697957[4] = (1.0 / 2.0) * (2 * X_Tx_ - 2 * sigma_p_Tx_ / (1 - omega_T_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * (Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_));
    }
    else {
        out_2760537359831697957[4] = (X_Tx_ - sigma_p_Tx_ / (1 - omega_T_)) / sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2));
    }
    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_2760537359831697957[5] = (1.0 / 2.0) * (2 * X_Ty_ - 2 * sigma_p_Ty_ / (1 - omega_T_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_2760537359831697957[5] = (1.0 / 2.0) * (2 * X_Ty_ - 2 * sigma_p_Ty_ / (1 - omega_T_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * (Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_));
    }
    else {
        out_2760537359831697957[5] = (X_Ty_ - sigma_p_Ty_ / (1 - omega_T_)) / sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2));
    }
    out_2760537359831697957[6] = 0;
    out_2760537359831697957[7] = 0;

}


void get_df_dEps(double X_Tx_, double X_Ty_, double Z_T_, double omega_N_, double omega_T_, double sigma_p_Tx_, double sigma_p_Ty_, double sigma_p_N_, double f_c0_, double f_c_, double f_s_, double f_t_, double m_, double* out_3788936598756039371) {

    out_3788936598756039371[0] = 0;
    out_3788936598756039371[1] = 0;
    out_3788936598756039371[2] = 0;
    out_3788936598756039371[3] = 0;
    out_3788936598756039371[4] = 0;
    out_3788936598756039371[5] = 0;
    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_3788936598756039371[6] = (1.0 / 2.0) * (2 * sigma_p_Tx_ * (-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_)) / pow(1 - omega_T_, 2) + 2 * sigma_p_Ty_ * (-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_)) / pow(1 - omega_T_, 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_3788936598756039371[6] = (1.0 / 2.0) * (2 * sigma_p_Tx_ * (-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_)) / pow(1 - omega_T_, 2) + 2 * sigma_p_Ty_ * (-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_)) / pow(1 - omega_T_, 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * (Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_));
    }
    else {
        out_3788936598756039371[6] = (sigma_p_Tx_ * (-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_)) / pow(1 - omega_T_, 2) + sigma_p_Ty_ * (-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_)) / pow(1 - omega_T_, 2)) / sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2));
    }
    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_3788936598756039371[7] = sigma_p_N_ * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) / (pow(f_t_, 2) * pow(-omega_N_ * ((sigma_p_N_ < 0) ? (
                                    0
                                    )
                                    : ((sigma_p_N_ == 0) ? (
                                        1.0 / 2.0
                                        )
                                        : (
                                            1
                                            ))) + 1, 2) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                0
                                                )
                                                : ((sigma_p_N_ == 0) ? (
                                                    1.0 / 2.0
                                                    )
                                                    : (
                                                        1
                                                        ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * pow(Z_T_ + f_s_ - f_t_ * m_, 2));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_3788936598756039371[7] = sigma_p_N_ * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) / (pow(f_c0_ - f_c_, 2) * pow(-omega_N_ * ((sigma_p_N_ < 0) ? (
                                    0
                                    )
                                    : ((sigma_p_N_ == 0) ? (
                                        1.0 / 2.0
                                        )
                                        : (
                                            1
                                            ))) + 1, 2) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                0
                                                )
                                                : ((sigma_p_N_ == 0) ? (
                                                    1.0 / 2.0
                                                    )
                                                    : (
                                                        1
                                                        ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * pow(Z_T_ + f_c_ * m_ + f_s_, 2));
    }
    else {
        out_3788936598756039371[7] = sigma_p_N_ * m_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) / pow(-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1, 2);
    }

}


void get_Phi(double S_N_, double S_T_, double X_Tx_, double X_Ty_, double Y_N_, double Y_T_, double Z_T_, double omega_N_, double omega_T_, double sigma_p_Tx_, double sigma_p_Ty_, double sigma_p_N_, double c_N_, double c_T_, double eta_, double f_c0_, double f_c_, double f_s_, double f_t_, double m_, double r_N_, double r_T_, double* out_7162032753439305039) {

    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_7162032753439305039[0] = (-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((1 - omega_T_) * (Z_T_ + f_s_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_7162032753439305039[0] = (-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((1 - omega_T_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * (Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_));
    }
    else {
        out_7162032753439305039[0] = (-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_)) / ((1 - omega_T_) * sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)));
    }
    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_7162032753439305039[1] = (-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((1 - omega_T_) * (Z_T_ + f_s_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_7162032753439305039[1] = (-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((1 - omega_T_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * (Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_));
    }
    else {
        out_7162032753439305039[1] = (-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_)) / ((1 - omega_T_) * sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)));
    }
    if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) {
        out_7162032753439305039[2] = (1.0 / 2.0) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (2 * omega_N_ * sigma_p_N_ * DiracDelta(sigma_p_N_) / pow(-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1, 2) + 2 / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                    0
                                    )
                                    : ((sigma_p_N_ == 0) ? (
                                        1.0 / 2.0
                                        )
                                        : (
                                            1
                                            ))) + 1)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                0
                                                )
                                                : ((sigma_p_N_ == 0) ? (
                                                    1.0 / 2.0
                                                    )
                                                    : (
                                                        1
                                                        ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * pow(Z_T_ + f_s_ - f_t_ * m_, 2));
    }
    else if ((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) {
        out_7162032753439305039[2] = (1.0 / 2.0) * (2 * omega_N_ * sigma_p_N_ * DiracDelta(sigma_p_N_) / pow(-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1, 2) + 2 / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1)) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                    0
                                    )
                                    : ((sigma_p_N_ == 0) ? (
                                        1.0 / 2.0
                                        )
                                        : (
                                            1
                                            ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                0
                                                )
                                                : ((sigma_p_N_ == 0) ? (
                                                    1.0 / 2.0
                                                    )
                                                    : (
                                                        1
                                                        ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * pow(Z_T_ + f_c_ * m_ + f_s_, 2));
    }
    else {
        out_7162032753439305039[2] = omega_N_ * sigma_p_N_ * m_ * DiracDelta(sigma_p_N_) / pow(-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1, 2) + m_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1);
    }
    out_7162032753439305039[3] = -(((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) ? (
                    -(2 * Z_T_ + 2 * f_s_ - 2 * f_t_ * m_) / (Z_T_ + f_s_ - 2 * f_t_ * m_) + (pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2)) / pow(Z_T_ + f_s_ - 2 * f_t_ * m_, 2) + (-m_ * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(-Z_T_ - f_s_ + 2 * f_t_ * m_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + (1.0 / 2.0) * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * (4 * Z_T_ + 4 * f_s_ - 4 * f_t_ * m_) * (pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2)) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) - (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 3)) - 1.0 / 2.0 * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * pow(Z_T_ + f_s_ - 2 * f_t_ * m_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) - 1.0 / 2.0 * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(Z_T_ + f_s_, 2) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + (1.0 / 2.0) * pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                    0
                                    )
                                    : ((sigma_p_N_ == 0) ? (
                                        1.0 / 2.0
                                        )
                                        : (
                                            1
                                            ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * (4 * Z_T_ + 4 * f_s_ - 4 * f_t_ * m_) * (pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2)) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) - pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                0
                                                )
                                                : ((sigma_p_N_ == 0) ? (
                                                    1.0 / 2.0
                                                    )
                                                    : (
                                                        1
                                                        ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 3))) / sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                            0
                                                            )
                                                            : ((sigma_p_N_ == 0) ? (
                                                                1.0 / 2.0
                                                                )
                                                                : (
                                                                    1
                                                                    ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)))
                    )
        : (((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) ? (
                        -(2 * Z_T_ + 2 * f_c_ * m_ + 2 * f_s_) / (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_) + (pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2)) / pow(Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_, 2) + (-m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                            0
                            )
                            : ((sigma_p_N_ == 0) ? (
                                1.0 / 2.0
                                )
                                : (
                                    1
                                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * pow(-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_, 2)) + (1.0 / 2.0) * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * (4 * Z_T_ + 4 * f_c_ * m_ + 4 * f_s_) * (pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2)) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) - 1.0 / 2.0 * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * pow(Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_, 2)) - (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 3) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) - 1.0 / 2.0 * (pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(Z_T_ + f_c0_ * m_ + f_s_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + (1.0 / 2.0) * (4 * Z_T_ + 4 * f_c_ * m_ + 4 * f_s_) * pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                        0
                                        )
                                        : ((sigma_p_N_ == 0) ? (
                                            1.0 / 2.0
                                            )
                                            : (
                                                1
                                                ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * (pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2)) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2)) - pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                    0
                                                    )
                                                    : ((sigma_p_N_ == 0) ? (
                                                        1.0 / 2.0
                                                        )
                                                        : (
                                                            1
                                                            ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 3))) / sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                                                                0
                                                                )
                                                                : ((sigma_p_N_ == 0) ? (
                                                                    1.0 / 2.0
                                                                    )
                                                                    : (
                                                                        1
                                                                        ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2)))
                        )
            : (
                -1
                )));
    out_7162032753439305039[4] = -(((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) ? (
                    (1.0 / 2.0) * (2 * X_Tx_ - 2 * sigma_p_Tx_ / (1 - omega_T_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))
                    )
        : (((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) ? (
                        (1.0 / 2.0) * (2 * X_Tx_ - 2 * sigma_p_Tx_ / (1 - omega_T_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                            0
                            )
                            : ((sigma_p_N_ == 0) ? (
                                1.0 / 2.0
                                )
                                : (
                                    1
                                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * (Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_))
                        )
            : (
                (X_Tx_ - sigma_p_Tx_ / (1 - omega_T_)) / sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))
                )));
    out_7162032753439305039[5] = -(((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
        0
        )
        : ((sigma_p_N_ == 0) ? (
            1.0 / 2.0
            )
            : (
                1
                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_)) * (-Z_T_ - f_s_ + 2 * f_t_ * m_) / (pow(f_t_, 2) * m_)) * (((f_t_) > 0) - ((f_t_) < 0)) * (((m_) > 0) - ((m_) < 0)) < 0) ? (
                    (1.0 / 2.0) * (2 * X_Ty_ - 2 * sigma_p_Ty_ / (1 - omega_T_)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / ((Z_T_ + f_s_) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                        0
                        )
                        : ((sigma_p_N_ == 0) ? (
                            1.0 / 2.0
                            )
                            : (
                                1
                                ))) + 1) - pow(f_t_, 2) * m_ / (-Z_T_ - f_s_ + 2 * f_t_ * m_), 2) * pow(pow(f_t_, 2) * pow(m_, 2) - 2 * f_t_ * m_ * (Z_T_ + f_s_) + pow(Z_T_ + f_s_, 2), 2) / (pow(f_t_, 2) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))) * (Z_T_ + f_s_ - 2 * f_t_ * m_) * pow(Z_T_ + f_s_ - f_t_ * m_, 2))
                    )
        : (((fabs(sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))) + (Z_T_ + f_c0_ * m_ + f_s_) * (sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
            0
            )
            : ((sigma_p_N_ == 0) ? (
                1.0 / 2.0
                )
                : (
                    1
                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_)) * (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_) / (m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)))) * (((m_) > 0) - ((m_) < 0)) * (((f_c0_ - f_c_) > 0) - ((f_c0_ - f_c_) < 0)) < 0) ? (
                        (1.0 / 2.0) * (2 * X_Ty_ - 2 * sigma_p_Ty_ / (1 - omega_T_)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (sqrt((pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2)) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / ((Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_)) + pow(sigma_p_N_ / (-omega_N_ * ((sigma_p_N_ < 0) ? (
                            0
                            )
                            : ((sigma_p_N_ == 0) ? (
                                1.0 / 2.0
                                )
                                : (
                                    1
                                    ))) + 1) + f_c0_ - m_ * (pow(f_c0_, 2) - 2 * f_c0_ * f_c_ + pow(f_c_, 2)) / (-Z_T_ + f_c0_ * m_ - 2 * f_c_ * m_ - f_s_), 2) * pow(pow(f_c0_, 2) * pow(m_, 2) - 2 * f_c0_ * f_c_ * pow(m_, 2) - 2 * f_c0_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(f_c_, 2) * pow(m_, 2) + 2 * f_c_ * m_ * (Z_T_ + f_c0_ * m_ + f_s_) + pow(Z_T_ + f_c0_ * m_ + f_s_, 2), 2) / (pow(f_c0_ - f_c_, 2) * pow(Z_T_ + f_c_ * m_ + f_s_, 2))) * (Z_T_ + f_c0_ * m_ + f_s_) * pow(Z_T_ + f_c_ * m_ + f_s_, 2) * (Z_T_ - f_c0_ * m_ + 2 * f_c_ * m_ + f_s_))
                        )
            : (
                (X_Ty_ - sigma_p_Ty_ / (1 - omega_T_)) / sqrt(pow(-X_Tx_ + sigma_p_Tx_ / (1 - omega_T_), 2) + pow(-X_Ty_ + sigma_p_Ty_ / (1 - omega_T_), 2))
                )));
    out_7162032753439305039[6] = pow(S_T_, -r_T_) * (pow(S_T_, r_T_) * eta_ * pow((Y_N_ + Y_T_) / (sqrt(S_N_) * sqrt(S_T_)), sqrt(r_N_) * sqrt(r_T_)) * pow(omega_N_ * omega_T_ - omega_N_ - omega_T_ + 1, (1.0 / 2.0) * sqrt(c_N_) * sqrt(c_T_)) - pow(Y_T_, r_T_) * eta_ * pow(1 - omega_T_, c_T_) + pow(Y_T_, r_T_) * pow(1 - omega_T_, c_T_));
    out_7162032753439305039[7] = pow(S_N_, -r_N_) * (pow(S_N_, r_N_) * eta_ * pow((Y_N_ + Y_T_) / (sqrt(S_N_) * sqrt(S_T_)), sqrt(r_N_) * sqrt(r_T_)) * pow(omega_N_ * omega_T_ - omega_N_ - omega_T_ + 1, (1.0 / 2.0) * sqrt(c_N_) * sqrt(c_T_)) - pow(Y_N_, r_N_) * eta_ * pow(1 - omega_N_, c_N_) + pow(Y_N_, r_N_) * pow(1 - omega_N_, c_N_));

}

//Eps = sp.Matrix([u_p_Tx, u_p_Ty, u_p_N, z_T, alpha_Tx, alpha_Ty, omega_T, omega_N])
//Sig = sp.Matrix([sigma_p_Tx, sigma_p_Ty, sigma_p_N, Z_T, X_Tx, X_Ty, Y_T, Y_N])



void get_Eps_k1(double u_Tx_, double u_Ty_, double u_N_, const double* Eps_n, double lam_k, const double* Sig_n,
    double* Eps_k, double* Sig_k, double* Phi_k,
    double E_N_, double E_T_,
    double K_T_, double gamma_T_,
    double S_N_, double S_T_,
    double X_Tx_, double X_Ty_, double Y_N_, double Y_T_, double Z_T_,
    double sigma_p_Tx_, double sigma_p_Ty_,
    double c_N_, double c_T_, double eta_,
    double f_c0_, double f_c_, double f_s_, double f_t_, double m_,
    double r_N_, double r_T_) {


    /* Python implementation
    def get_Eps_k1(u_Tx_n1, u_Ty_n1, u_N_n1, Eps_n, lam_k, Sig_n, Eps_k, **kw):
        Sig_k = get_Sig(u_Tx_n1, u_Ty_n1, u_N_n1, Sig_n, Eps_k, **kw)[0]
        Phi_k = get_Phi(Eps_k, Sig_k, **kw)
        Eps_k1 = Eps_n + lam_k * Phi_k[:,0]
        return Eps_k1
    */

    //                     0           1             2           3          4       5        6     7
    //double Eps_n[8] = { u_p_Tx    , u_p_Ty    , u_p_N  ,     z_T , alpha_Tx, alpha_Ty, omega_T, omega_N };
    //double Sig_n[8] = { sigma_p_Tx, sigma_p_Ty, sigma_p_N  , Z_T     , X_Tx    , X_Ty   , Y_T,       Y_N };


    // update Sig_k
    double u_p_Tx_ = Eps_k[0];
    double u_p_Ty_ = Eps_k[1];
    double u_p_N_ = Eps_k[2];
    double z_T_ = Eps_k[3];
    double alpha_Tx_ = Eps_k[4];
    double alpha_Ty_ = Eps_k[5];
    double omega_T_ = Eps_k[6];
    double omega_N_ = Eps_k[7];
    double sigma_p_N_ = Sig_n[2];
    get_Sig(u_Ty_, E_N_, E_T_, K_T_, alpha_Tx_, alpha_Ty_, gamma_T_, omega_N_, omega_T_, sigma_p_N_, u_p_N_, u_p_Tx_, u_p_Ty_, u_N_, u_Tx_, z_T_, Sig_k);


    // update Phi_k
    sigma_p_Tx_ = Sig_k[0];
    sigma_p_Ty_ = Sig_k[1];
    sigma_p_N_ = Sig_k[2];
    Z_T_ = Sig_k[3];
    X_Tx_ = Sig_k[4];
    X_Ty_ = Sig_k[5];
    Y_T_ = Sig_k[6];
    Y_N_ = Sig_k[7];

    get_Phi(S_N_, S_T_, X_Tx_, X_Ty_, Y_N_, Y_T_, Z_T_, omega_N_, omega_T_, sigma_p_Tx_, sigma_p_Ty_, sigma_p_N_, c_N_, c_T_, eta_, f_c0_, f_c_, f_s_, f_t_, m_, r_N_, r_T_, Phi_k);

    for (int i = 0; i < 8; i++) //[u_p_Tx, u_p_Ty, u_p_N, z_T, alpha_Tx, alpha_Ty, omega_T, omega_N]
        Eps_k[i] = Eps_n[i] + lam_k * Phi_k[i];
}


double get_f_df(double u_Tx_, double u_Ty_, double u_N_, const double* Sig_n, const double* Eps_k,
    double* Sig_k, double* dSig_dEps_k, double* df_dSig_k, double* ddf_dEps_k, double* Phi_k, double* df_dlambda,
    double E_N_, double E_T_,
    double K_T_, double gamma_T_,
    double S_N_, double S_T_,
    double X_Tx_, double X_Ty_, double Y_N_, double Y_T_, double Z_T_,
    double sigma_p_Tx_, double sigma_p_Ty_,
    double c_N_, double c_T_, double eta_,
    double f_c0_, double f_c_, double f_s_, double f_t_, double m_,
    double r_N_, double r_T_) {

    /* Python implementation
    def get_f_df(u_Tx_n1, u_Ty_n1, u_N_n1, Sig_n, Eps_k, **kw):
        Sig_k = get_Sig(u_Tx_n1, u_Ty_n1, u_N_n1, Sig_n, Eps_k, **kw)[0]
        dSig_dEps_k = get_dSig_dEps(u_Tx_n1, u_Ty_n1, u_N_n1, Sig_n, Eps_k, **kw)
        f_k = np.array([get_f(Eps_k, Sig_k, **kw)])
        if f_k < -1e-1: # elastic step - should be fraction of current strength
            return f_k, np.array([[0]]), Sig_k
        df_dSig_k = get_df_dSig(Eps_k, Sig_k, **kw)
        ddf_dEps_k = get_ddf_dEps(Eps_k, Sig_k, **kw)
        df_dEps_k = np.einsum(
            'ik,ji->jk', df_dSig_k, dSig_dEps_k) + ddf_dEps_k
        Phi_k = get_Phi(Eps_k, Sig_k, **kw)
        dEps_dlambda_k = Phi_k
        df_dlambda = np.einsum(
            'ki,kj->ij', df_dEps_k, dEps_dlambda_k)
        df_k = df_dlambda
        return f_k, df_k, Sig_k
    */


    // accept Eps_k
    double u_p_Tx_ = Eps_k[0];
    double u_p_Ty_ = Eps_k[1];
    double u_p_N_ = Eps_k[2];
    double z_T_ = Eps_k[3];
    double alpha_Tx_ = Eps_k[4];
    double alpha_Ty_ = Eps_k[5];
    double omega_T_ = Eps_k[6];
    double omega_N_ = Eps_k[7];

    double sigma_p_N_ = Sig_n[2]; // for the Heaviside
    get_Sig(u_Ty_, E_N_, E_T_, K_T_, alpha_Tx_, alpha_Ty_, gamma_T_, omega_N_, omega_T_, sigma_p_N_, u_p_N_, u_p_Tx_, u_p_Ty_, u_N_, u_Tx_, z_T_, Sig_k);

    // update dSig_dEps_k
    get_dSig_dEps(u_Ty_, E_N_, E_T_, K_T_, gamma_T_, omega_N_, omega_T_, sigma_p_N_, u_p_N_, u_p_Tx_, u_p_Ty_, u_N_, u_Tx_, dSig_dEps_k);

    //                     0           1             2           3          4       5        6     7
    //double Eps_n[8] = { u_p_Tx    , u_p_Ty    , u_p_N  ,     z_T , alpha_Tx, alpha_Ty, omega_T, omega_N };
    //double Sig_n[8] = { sigma_p_Tx, sigma_p_Ty, sigma_p_N  , Z_T     , X_Tx    , X_Ty   , Y_T,       Y_N };

    sigma_p_Tx_ = Sig_k[0];
    sigma_p_Ty_ = Sig_k[1];
    sigma_p_N_ = Sig_k[2];
    Z_T_ = Sig_k[3];
    X_Tx_ = Sig_k[4];
    X_Ty_ = Sig_k[5];
    Y_T_ = Sig_k[6];
    Y_N_ = Sig_k[7];

    double f_k = get_f(X_Tx_, X_Ty_, Z_T_, omega_N_, omega_T_, sigma_p_Tx_, sigma_p_Ty_, sigma_p_N_, f_c0_, f_c_, f_s_, f_t_, m_);

    //TODO: check why not returning upon noticing f_k < 0
    if (f_k < -0.1) { //elastic step - should be fraction of current strength
        return f_k;
    }

    get_df_dSig(X_Tx_, X_Ty_, Z_T_, omega_N_, omega_T_, sigma_p_Tx_, sigma_p_Ty_, sigma_p_N_, f_c0_, f_c_, f_s_, f_t_, m_, df_dSig_k);
    get_df_dEps(X_Tx_, X_Ty_, Z_T_, omega_N_, omega_T_, sigma_p_Tx_, sigma_p_Ty_, sigma_p_N_, f_c0_, f_c_, f_s_, f_t_, m_, ddf_dEps_k);

    double df_dEps_k[8];

    //df_dEps_k = dSig_dEps_k*df_dSig_k + ddf_dEps_k
    for (int i = 0; i < 8; i++) {
        df_dEps_k[i] = ddf_dEps_k[i];
        for (int j = 0; j < 8; j++) df_dEps_k[i] += dSig_dEps_k[i * 8 + j] * df_dSig_k[j];
    }

    // update Phi_k
    get_Phi(S_N_, S_T_, X_Tx_, X_Ty_, Y_N_, Y_T_, Z_T_, omega_N_, omega_T_, sigma_p_Tx_, sigma_p_Ty_, sigma_p_N_, c_N_, c_T_, eta_, f_c0_, f_c_, f_s_, f_t_, m_, r_N_, r_T_, Phi_k);


    *df_dlambda = 0;
    for (int i = 0; i < 8; i++) //dot product
        *df_dlambda += df_dEps_k[i] * Phi_k[i];


    return f_k;
}


//NOTE: I included the stresses and displacements of the interface as state variables. This might not be correct, depending on the interface (of the interface model)
#define UserStateVarsCount 22 
char* UserStateVarNames[UserStateVarsCount] =
{
    "z_T",
    "Z_T",
    "alpha_N",
    "alpha_Tx",
    "alpha_Ty",
    "X_N",
    "X_Tx",
    "X_Ty",
    "Y_N",
    "Y_T",
    "omega_N",
    "omega_T",
    "u_p_N",
    "u_p_Tx",
    "u_p_Ty",
    "Plastic N compressive displacement",
    "Plastic N opening",
    "Plastic T slip",
    "Plastic T energy",
    "Damage T energy",
    "Plastic N energy",
    "Damage N energy",
};



/** Largest real number (value from the floating point library)
 *
 *  Not a compulsory part of the interface - the user may implement
 *  in another way.
 */
 //#define REAL_MAX DBL_MAX
  /** Minimum accepted divisor used to prevent division by zero/overflow
   * (based on a value from the floating point library)
   *
   *  Not a compulsory part of the interface - the user may implement
   *  in another way.
   */
   //#define MIN_DIV DBL_EPSILON*1000


	 // forward declaration (needed in calculate_response)

__declspec(dllexport) UINT CDECL UserDLLTangentStiff(double TangentMatrix[6 * 6],
	double E, double mu, double UserMaterialParams[], double UserMaterialState[]);

__declspec(dllexport) UINT CDECL UserDLLSecantStiff(double SecantMatrix[6 * 6],
	double E, double mu, double UserMaterialParams[], double UserMaterialState[]);

__declspec(dllexport) UINT CDECL UserDLLCalculateResponse(
	double deps[], /**< strain increment tensor stored as a vector,
					*   first the diagonal terms, followed by the off-diagonal
					*/
	double teps[], ///< total strain tensor (stored as a vector)
	double sigma[], ///< [in+out] stress tensor stored as a vector (as for deps)
	double E, ///< elastic modulus
	double mu, ///< Poisson's ratio
	double UserMaterialParams[], ///< vector of user material parameter values
	double UserMaterialState[] ///< [in+out] vector of user material state variables in the material point being calculated 

) {
	// get the state variables and material parameters

    double E_N = UserMaterialParams[0];
    double E_T = UserMaterialParams[1];
    double gamma_N = UserMaterialParams[2];
    double gamma_T = UserMaterialParams[3];
    double K_T = UserMaterialParams[4];
    double S_N = UserMaterialParams[5];
    double S_T = UserMaterialParams[6];
    double f_c0 = UserMaterialParams[7];
    double f_c = UserMaterialParams[8];
    double f_s = UserMaterialParams[9];
    double f_t = UserMaterialParams[10];
    double c_N = UserMaterialParams[11];
    double r_N = UserMaterialParams[12];
    double c_T = UserMaterialParams[13];
    double r_T = UserMaterialParams[14];
    double m = UserMaterialParams[15];
    double eta = UserMaterialParams[16];

    double sigma_p_N = sigma[0];
    double sigma_p_Tx = sigma[1];
    double sigma_p_Ty = sigma[2];
    double u_N = teps[3];   //NOTE: our material model is formulated in terms of stresses and displacements!
    double u_Tx = teps[4];
    double u_Ty = teps[5];

    double z_T = UserMaterialState[0];
    double Z_T = UserMaterialState[1];
    double alpha_N = UserMaterialState[2];
    double alpha_Tx = UserMaterialState[3];
    double alpha_Ty = UserMaterialState[4];
    double X_N = UserMaterialState[5];
    double X_Tx = UserMaterialState[6];
    double X_Ty = UserMaterialState[7];
    double Y_N = UserMaterialState[8];
    double Y_T = UserMaterialState[9];
    double omega_N = UserMaterialState[10];
    double omega_T = UserMaterialState[11];
    double u_p_N = UserMaterialState[12];
    double u_p_Tx = UserMaterialState[13];
    double u_p_Ty = UserMaterialState[14];
    double disp_compres_plastic = UserMaterialState[15];
    double disp_opening_plastic = UserMaterialState[16];
    double disp_shear_plastic = UserMaterialState[17];
    double plast_T_disip = UserMaterialState[18];
    double damage_T_disip = UserMaterialState[19];
    double plast_N_disip = UserMaterialState[20];
    double damage_N_disip = UserMaterialState[21];

    //TODO: put the allocation for these arrays to constructor of the SLIDE class
    double dSig_dEps_k[8 * 8];
    double df_dSig_k[8];
    double ddf_dEps_k[8];
    double Phi_k[8]; //= new double[8]; //double Phi_k[8];
    double Eps_n[8];
    double Sig_n[8];
    double Eps_k[8]; //= new double[8];
    double Sig_k[8]; //= new double[8];

    double f_k;
    double df_dlambda;

    Eps_n[0] = u_p_Tx;
    Eps_n[1] = u_p_Ty;
    Eps_n[2] = u_p_N;
    Eps_n[3] = z_T;
    Eps_n[4] = alpha_Tx;
    Eps_n[5] = alpha_Ty;
    Eps_n[6] = omega_T;
    Eps_n[7] = omega_N;
    //double Eps_n[8] = {u_p_Tx, u_p_Ty, u_p_N, z_T, alpha_Tx, alpha_Ty, omega_T, omega_N };


    Sig_n[0] = sigma_p_Tx;
    Sig_n[1] = sigma_p_Ty;
    Sig_n[2] = sigma_p_N;
    Sig_n[3] = Z_T;
    Sig_n[4] = X_Tx;
    Sig_n[5] = X_Ty;
    Sig_n[6] = Y_T;
    Sig_n[7] = Y_N;
    //double Sig_n[8] = { sigma_p_Tx , sigma_p_Ty, sigma_p_N, Z_T, X_Tx, X_Ty, Y_T, Y_N };

    //accept the previous state as initial iteration (8 entries from "n" to "t")
    std::memcpy(Eps_k, Eps_n, 8 * sizeof(double));
    std::memcpy(Sig_k, Sig_n, 8 * sizeof(double));
    //for (int i = 0; i < 8; i++) {
     //   Eps_k[i] = Eps_n[i];
     //   Sig_k[i] = Sig_n[i];
    //}

    /* Python code
        def get_material_model(u_Tx_n1, u_Ty_n1, u_N_n1, Sig_n, Eps_n, k_max, **kw):
        Eps_k = np.copy(Eps_n)
        Sig_k = np.copy(Sig_n)
        lam_k = 0
        f_k, df_k, Sig_k = get_f_df(u_Tx_n1, u_Ty_n1, u_N_n1, Sig_k, Eps_k, **kw)
        f_k_norm = np.linalg.norm(f_k)
        f_k_trial = f_k[0]
        dlam = 0
        k = 0
        while k < k_max:
            if (f_k_trial < 0 or f_k_norm < 1e-6) and np.fabs(dlam) < 1e-3:
                return Eps_k, Sig_k, k+1
            dlam = np.linalg.solve(df_k, -f_k)
            lam_k += dlam
            Eps_k = get_Eps_k1(u_Tx_n1, u_Ty_n1, u_N_n1, Eps_n, lam_k, Sig_k, Eps_k, **kw)
            f_k, df_k, Sig_k = get_f_df(u_Tx_n1, u_Ty_n1, u_N_n1, Sig_k, Eps_k, **kw)
            f_k_norm = np.linalg.norm(f_k)
            k += 1
        else:
            raise ValueError('no convergence')
    */

    double lam_k = 0.0;
    df_dlambda = 0.0;
    f_k = get_f_df(u_Tx, u_Ty, u_N, Sig_n, Eps_k,
        Sig_k, dSig_dEps_k, df_dSig_k, ddf_dEps_k, Phi_k, &df_dlambda,
        E_N, E_T,
        K_T, gamma_T,
        S_N, S_T,
        X_Tx, X_Ty, Y_N, Y_T, Z_T,
        sigma_p_Tx, sigma_p_Ty,
        c_N, c_T, eta,
        f_c0, f_c, f_s, f_t, m,
        r_N, r_T);

    double f_k_norm = abs(f_k);
    double f_k_trial = f_k;
    int k_max = 100;
    double dlam = 0.0;
    for (int k = 0; k < k_max; k++) {
        if (((f_k_trial < 0.0) || (f_k_norm < 1.0e-5)) && (abs(dlam) < 1.0e-5)) {
            //accept the k-th iteration

            sigma[0] = Sig_k[2];//sigma_p_N
            sigma[1] = Sig_k[0];//sigma_p_Tx 
            sigma[2] = Sig_k[1];//sigma_p_Ty

            // state variables
            UserMaterialState[0] = Eps_k[3];//z_T
            UserMaterialState[1] = Sig_k[3];//Z_T 
            UserMaterialState[2] = 0;       //alpha_N //Todo
            UserMaterialState[3] = Eps_k[4];//alpha_Tx
            UserMaterialState[4] = Eps_k[5];//alpha_Ty
            UserMaterialState[5] = 0;//X_N
            UserMaterialState[6] = Sig_k[4];//X_Tx
            UserMaterialState[7] = Sig_k[5];//X_Ty
            UserMaterialState[8] = Sig_k[7];//Y_N 
            UserMaterialState[9] = Sig_k[6];//Y_T 
            UserMaterialState[10] = Eps_k[7];//omega_N 
            UserMaterialState[11] = Eps_k[6];//omega_T 

            UserMaterialState[12] = Eps_k[2];//u_p_N
            UserMaterialState[13] = Eps_k[0];//u_p_Tx
            UserMaterialState[14] = Eps_k[1];//u_p_Ty

            // additional data for graphical postprocessing
            UserMaterialState[15] = (stv[35] < 0) ? stv[35] : 0; //disp_compres_plastic;
            UserMaterialState[16] = (stv[35] > 0) ? stv[35] : 0; //disp_opening_plastic;
            UserMaterialState[17] = sqrt(u_p_Tx * u_p_Tx + u_p_Ty * u_p_Ty); //disp_shear_plastic (Euclidean norm of two orthogonal plastic displacements);
            
            // evaluation of energy dissipation

            UserMaterialState[18] += sqrt(Sig_k[0] * Sig_k[0] + Sig_k[1] * Sig_k[1]) * (sqrt(Eps_k[0] * Eps_k[0] + Eps_k[1] * Eps_k[1]) - sqrt(Eps_n[0] * Eps_n[0] + Eps_n[1] * Eps_n[1])) -
                Sig_k[3] * (Eps_k[3] - Eps_n[3]) -
                sqrt(Eps_k[4] * Sig_k[4] + Sig_k[5] * Sig_k[5]) * (sqrt(Eps_k[4] * Eps_k[4] + Eps_k[5] * Eps_k[5]) - sqrt(Eps_n[4] * Eps_n[4] + Eps_n[5] * Eps_n[5]));   //plast_T_disip;
            UserMaterialState[19] += Sig_k[6] * (Eps_k[6] - Eps_n[6]);   //damage_T_disip;
            UserMaterialState[20] += Sig_k[2] * (Eps_k[2] - Eps_n[2]);   //plast_N_disip;
            UserMaterialState[21] += Sig_k[7] * (Eps_k[7] - Eps_n[7]);   //damage_N_disip;

            //cout << "Slide convergence" << endl;
            //cout << Sig_k[2] << endl;
            return;
        }
        //else {
        dlam = -f_k / df_dlambda;
        lam_k += dlam;
        get_Eps_k1(u_Tx, u_Ty, u_N, Eps_n, lam_k, Sig_n,
            Eps_k, Sig_k, Phi_k,
            E_N, E_T,
            K_T, gamma_T,
            S_N, S_T,
            X_Tx, X_Ty, Y_N, Y_T, Z_T,
            sigma_p_Tx, sigma_p_Ty,
            c_N, c_T, eta,
            f_c0, f_c, f_s, f_t, m,
            r_N, r_T);
        f_k = get_f_df(u_Tx, u_Ty, u_N, Sig_n, Eps_k,
            Sig_k, dSig_dEps_k, df_dSig_k, ddf_dEps_k, Phi_k, &df_dlambda,
            E_N, E_T,
            K_T, gamma_T,
            S_N, S_T,
            X_Tx, X_Ty, Y_N, Y_T, Z_T,
            sigma_p_Tx, sigma_p_Ty,
            c_N, c_T, eta,
            f_c0, f_c, f_s, f_t, m,
            r_N, r_T);

        f_k_norm = abs(f_k);

    }
    //cout << "Slide no convergence" << endl;
    return 0;
}


__declspec(dllexport) UINT CDECL UserDLLResetState(
	double E, ///< Young modulus
	double mu, ///< Poisson's ratio 
	double UserMaterialParams[], ///< user material parameters array
	double UserMaterialState[] ///< [out] user state variables array
) {
	int i;
	for (i = 0; i < UserStateVarsCount; i++) {
		UserMaterialState[i] = 0.0;
	}

	//calculate_MPNN();
	return 0;
}


__declspec(dllexport) UINT CDECL UserDLLTangentStiff(
	double TangentMatrix[6 * 6], ///< [out] the local tangential stiffness matrix as a vector
	double E, ///< Young modulus
	double mu, ///< Poisson's ratio
	double UserMaterialParams[], ///< user material parameters array
	double UserMaterialState[] ///< user state variables array


) {
    //Implementation not tested, just a sketch of how it could be done! (we implemented the model for an explicit solver)

    void get_material_model(double u1, double u2, double u3, double du1, double du2, double du3, double Sig_k[], double Eps_k[], int k_max, double E_T_, double gamma_T_, double K_T_, double S_T_, double c_T_, double f_s_, double E_N_, double S_N_, double c_N_, double m_, double f_t_, double f_c_, double f_c0_, double r_N_, double r_T_, double eta_, double Eps_k1[], double Sig_k1[]) {
        // This function is defined at UserDLLCalculateResponse
    }

    void compute_numerical_tangent(double u_loc[], double du_loc[], double Sig_k[], double Eps_k[], double C[][3], double du_) {
        double Sig_kk[3], Sig_k1[3];
        int k_max = 0; // assuming k_max is initialized to 0

        // Computing the numerical tangent, first direction
        UserDLLCalculateResponse(u_loc[0] - du_, u_loc[1], u_loc[2], du_loc[0], du_loc[1], du_loc[2], Sig_k, Eps_k, k_max, E_T_, gamma_T_, K_T_, S_T_, c_T_, f_s_, E_N_, S_N_, c_N_, m_, f_t_, f_c_, f_c0_, r_N_, r_T_, eta_, Eps_k1, Sig_k1);
        TangentMatrix[0][0] = (Sig_kk[0] - Sig_k1[0]) / du_ * 0.5;
        TangentMatrix[1][0] = (Sig_kk[1] - Sig_k1[1]) / du_ * 0.5;
        TangentMatrix[2][0] = (Sig_kk[2] - Sig_k1[2]) / du_ * 0.5;

        // Computing the numerical tangent, second direction
        UserDLLCalculateResponse(u_loc[0], u_loc[1] - du_, u_loc[2], du_loc[0], du_loc[1], du_loc[2], Sig_k, Eps_k, k_max, E_T_, gamma_T_, K_T_, S_T_, c_T_, f_s_, E_N_, S_N_, c_N_, m_, f_t_, f_c_, f_c0_, r_N_, r_T_, eta_, Eps_k1, Sig_k1);
        TangentMatrix[0][1] = (Sig_kk[0] - Sig_k1[0]) / du_ * 0.5;
        TangentMatrix[1][1] = (Sig_kk[1] - Sig_k1[1]) / du_ * 0.5;
        TangentMatrix[2][1] = (Sig_kk[2] - Sig_k1[2]) / du_ * 0.5;

        // Computing the numerical tangent, third direction
        UserDLLCalculateResponse(u_loc[0], u_loc[1], u_loc[2] - du_, du_loc[0], du_loc[1], du_loc[2], Sig_k, Eps_k, k_max, E_T_, gamma_T_, K_T_, S_T_, c_T_, f_s_, E_N_, S_N_, c_N_, m_, f_t_, f_c_, f_c0_, r_N_, r_T_, eta_, Eps_k1, Sig_k1);
        TangentMatrix[0][2] = (Sig_kk[0] - Sig_k1[0]) / du_ * 0.5;
        TangentMatrix[1][2] = (Sig_kk[1] - Sig_k1[1]) / du_ * 0.5;
        TangentMatrix[2][2] = (Sig_kk[2] - Sig_k1[2]) / du_ * 0.5;
    }

	return 0;
}


__declspec(dllexport) UINT CDECL UserDLLElasticStiff(
	double ElasticMatrix[6 * 6], ///< [in, out] the local elastic stiffness matrix as a vector
	double E, ///< Young modulus
	double mu, ///< Poisson's ratio
	double UserMaterialParams[] ///< user material parameters array
 /**
  * \retval 0 OK
  * \retval nonzero error
  */
) {
	return 0; // use the inherited elastic matrix unchanged
}


__declspec(dllexport) UINT CDECL UserDLLSecantStiff(
	double SecantMatrix[6 * 6], ///< [out] the local tangential stiffness matrix as a vector
	double E, ///< Young modulus
	double mu, ///< Poisson's ratio
	double UserMaterialParams[], ///< user material parameters array
	double UserMaterialState[] ///< user state variables array

) {
    //Implementation not tested, just a sketch of how it could be done! (we implemented the model for an explicit solver)
    SecantMatrix[0][0] = E_N * (1 - omega_N);
    SecantMatrix[1][1] = E_T * (1 - omega_Tx);
    SecantMatrix[2][2] = E_T * (1 - omega_Ty);

	return 0;
}

/**
 *
 * \brief Purpose: transformation of coordinate system due to large deformations.
 *
 */
__declspec(dllexport) UINT CDECL UserDLLTransformState(
	double DefGradient[], ///< the deformation gradient matrix based used for transformation of the elastic stresses and strains
	double eps[], ///< [in+out] total strain tensor (stored as a vector)
	double sigma[], ///< [in+out] stress tensor stored as a vector (as for deps)
	double E, ///< Young modulus
	double mu, ///< Poisson's ratio
	double UserMaterialParams[], ///< user material parameters array
	double UserMaterialState[] ///< [in+out] user state variables array

) {
	return 0;
}

/**
 *
 * \brief Purpose: The number of user defined material parameters.
 *
 * The user has to define this function to let the ATENA kernel know how many
 * additional parameters the material has, which is required among others
 * when reading the material definition with the parameter values from an input file.
 *
 * @return  the number of additional user material parameters
 *
 */
__declspec(dllexport) UINT CDECL UserDLLMaterialParamsCount() {
	return UserMaterialParamsCount; // the number of additional user material parameters
}

/**
 *
 * \brief Purpose: The number of user defined material state variables.
 *
 * The user has to define this function to let the ATENA kernel know how many
 * additional state variables the material has in each material point,
 * which is required among others when offering the list of quantities
 * available for postprocessing.
 *
 * @return  the number of additional user material state variables
 *
 */
__declspec(dllexport) UINT CDECL UserDLLStateVarsCount() {
	return UserStateVarsCount; // the number of additional user state variables
}

/**
 *
 * \brief Purpose: The name of a user defined material parameter.
 *
 * The user has to define this function to let the ATENA kernel know the
 * names of the additional parameters the material has, which is required among others
 * when reading the material definition with the parameter values from an input file.
 *
 * @param MaterialParamNo  parameter number (id) 0..UserMaterialParamsCount-1
 *
 * \retval string name of the MaterialParamNo-th user material parameter
 * \retval NULL invalid parameter number (out of range)
 *
 */
__declspec(dllexport) LPSTR CDECL UserDLLMaterialParamName(UINT MaterialParamNo) {
	if ((MaterialParamNo >= 0) && (MaterialParamNo < UserMaterialParamsCount)) {
		return UserMaterialParamNames[MaterialParamNo];
	}
	else {
		return NULL; // invalid material parameter id
	}
}

/**
 *
 * \brief Purpose: The name of a user defined material state variable.
 *
 * The user has to define this function to let the ATENA kernel know the
 * names of the additional state variables the material has,
 * which is required among others when offering the list of quantities
 * available for postprocessing.
 *
 * @param StateVarNo  parameter number (id) 0..UserStateVarsCount-1
 *
 * \retval string name of the StateVarNo-th user material state variable
 * \retval NULL invalid parameter number (out of range)
 *
 */
__declspec(dllexport) LPSTR CDECL UserDLLStateVarName(UINT StateVarNo) {
	if ((StateVarNo >= 0) && (StateVarNo < UserStateVarsCount)) {
		return UserStateVarNames[StateVarNo];
	}
	else {
		return NULL; // invalid state variable id
	}
}
/** @} */


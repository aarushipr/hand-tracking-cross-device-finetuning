/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@file
@brief Uniform Bezier curve for euclidean vectors
*/

#pragma once

#include <basalt/spline/spline_common.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/sophus_utils.hpp>

#include <Eigen/Dense>

#include <array>
#include "spline_common.h"

namespace basalt {

/// @brief Uniform Bezier curve for euclidean vectors with dimention DIM of
/// order
/// N
template <int _DIM, int _N, typename _Scalar = double>
class RdBezier {
 public:
  static constexpr int N = _N;        ///< Order of the spline.
  static constexpr int DEG = _N - 1;  ///< Degree of the spline.

  static constexpr int DIM = _DIM;  ///< Dimension of euclidean vector space.

  static constexpr _Scalar NS_TO_S = 1e-9;  ///< Nanosecond to second conversion
  static constexpr _Scalar S_TO_NS = 1e9;   ///< Second to nanosecond conversion

  using MatN = Eigen::Matrix<_Scalar, _N, _N>;
  using VecN = Eigen::Matrix<_Scalar, _N, 1>;

  using VecD = Eigen::Matrix<_Scalar, _DIM, 1>;
  using MatD = Eigen::Matrix<_Scalar, _DIM, _DIM>;

  /// @brief Struct to store the Jacobian of the Bezier curve
  struct JacobianStruct {
    size_t
        start_idx;  ///< Start index of the non-zero elements of the Jacobian.
    std::array<_Scalar, N> d_val_d_knot;  ///< Value of nonzero Jacobians.
  };

  struct IntegralJacobianStruct {
    std::array<VecD, _N> d_int_d_knot;
  };

  /// @brief Default constructor
  RdBezier() = default;

  /// @brief Constructor with knot interval and start time
  ///
  /// @param[in] time_interval_ns knot time interval in nanoseconds
  /// @param[in] start_time_ns start time of the spline in nanoseconds
  RdBezier(int64_t time_interval_ns, int64_t start_time_ns = 0)
      : dt_ns_(time_interval_ns), start_t_ns_(start_time_ns) {
    pow_inv_dt_[0] = 1.0;
    pow_inv_dt_[1] = S_TO_NS / dt_ns_;

    for (size_t i = 2; i < N; i++) {
      pow_inv_dt_[i] = pow_inv_dt_[i - 1] * pow_inv_dt_[1];
    }
  }

  /// @brief Cast to different scalar type
  template <typename Scalar2>
  inline RdBezier<_DIM, _N, Scalar2> cast() const {
    RdBezier<_DIM, _N, Scalar2> res;

    res.dt_ns_ = dt_ns_;
    res.start_t_ns_ = start_t_ns_;

    for (int i = 0; i < _N; i++) {
      res.pow_inv_dt_[i] = pow_inv_dt_[i];
    }

    for (int i = 0; i < N; i++) {
      res.knots_[i] = knots_[i].template cast<Scalar2>();
    }

    return res;
  }

  /// @brief Set start time for spline
  ///
  /// @param[in] start_time_ns start time of the spline in nanoseconds
  inline void setStartTimeNs(int64_t start_time_ns) {
    start_t_ns_ = start_time_ns;
  }

  /// @brief Maximum time represented by spline
  ///
  /// @return maximum time represented by spline in nanoseconds
  int64_t maxTimeNs() const {
    return start_t_ns_ + (knots_.size() - N + 1) * dt_ns_ - 1;
  }

  /// @brief Minimum time represented by spline
  ///
  /// @return minimum time represented by spline in nanoseconds
  int64_t minTimeNs() const { return start_t_ns_; }

  /// @brief Gererate random trajectory
  ///
  /// @param[in] static_init if true the first N knots will be the same
  /// resulting in static initial condition
  void genRandomTrajectory(bool static_init = false) {
    if (static_init) {
      VecD rnd = VecD::Random() * 5;

      for (int i = 0; i < N; i++) {
        knots_[i] = rnd;
      }
    } else {
      for (int i = 0; i < N; i++) {
        knots_[i] = VecD::Random();
      }
    }
  }

  /// @brief Return the first knot of the spline
  ///
  /// @return first knot of the spline
  inline const VecD& knotsFront() const { return knots_.front(); }

  /// @brief Return reference to the knot with index i
  ///
  /// @param i index of the knot
  /// @return reference to the knot
  inline VecD& getKnot(int i) { return knots_[i]; }

  /// @brief Return const reference to the knot with index i
  ///
  /// @param i index of the knot
  /// @return const reference to the knot
  inline const VecD& getKnot(int i) const { return knots_[i]; }

  /// @brief Return time interval in nanoseconds
  ///
  /// @return time interval in nanoseconds
  int64_t getTimeIntervalNs() const { return dt_ns_; }

  /// @brief Evaluate value or derivative of the spline
  ///
  /// @param Derivative derivative to evaluate (0 for value)
  /// @param[in] time_ns time for evaluating of the spline in nanoseconds
  /// @param[out] J if not nullptr, return the Jacobian of the value with
  /// respect to knots
  /// @return value of the spline or derivative. Euclidean vector of dimention
  /// DIM.
  template <int Derivative = 0>
  VecD evaluate(int64_t time_ns, JacobianStruct* J = nullptr) const {
    int64_t st_ns = (time_ns - start_t_ns_);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns_);

    int64_t s = st_ns / dt_ns_;
    double u = double(st_ns % dt_ns_) / double(dt_ns_);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(
        size_t(s + N) <= knots_.size(),
        "s " << s << " N " << N << " knots.size() " << knots_.size());

    VecN p;
    baseCoeffsWithTime<Derivative>(p, u);

    VecN coeff = pow_inv_dt_[Derivative] * (BLENDING_MATRIX * p);

    // std::cerr << "p " << p.transpose() << std::endl;
    // std::cerr << "coeff " << coeff.transpose() << std::endl;

    VecD res;
    res.setZero();

    for (int i = 0; i < N; i++) {
      res += coeff[i] * knots_[s + i];

      if (J) {
        J->d_val_d_knot[i] = coeff[i];
      }
    }

    if (J) {
      J->start_idx = s;
    }

    return res;
  }

  /// @brief Alias for first derivative of spline. See \ref evaluate.
  inline VecD velocity(int64_t time_ns, JacobianStruct* J = nullptr) const {
    return evaluate<1>(time_ns, J);
  }

  /// @brief Alias for second derivative of spline. See \ref evaluate.
  inline VecD acceleration(int64_t time_ns, JacobianStruct* J = nullptr) const {
    return evaluate<2>(time_ns, J);
  }

  /// @brief Evaluate integral of the squared value or squared time derivative
  /// of the spline
  template <int Derivative>
  inline _Scalar evaluateIntegralSquared(
      IntegralJacobianStruct* J = nullptr) const {
    _Scalar res = _Scalar(0);

    if (Derivative >= 0 && Derivative < N) {
      Eigen::Matrix<double, DIM, N> knots_matrix;
      Eigen::Matrix<double, DIM, N> tmp;

      for (int i = 0; i < N; i++) {
        knots_matrix.col(i) = getKnot(i);
      }

      _Scalar scaling;
      if (Derivative == 0) {
        scaling = _Scalar(1.0) / pow_inv_dt_[1];
      } else {
        scaling = pow_inv_dt_[Derivative] * pow_inv_dt_[Derivative - 1];
      }

      tmp = knots_matrix * BLENDING_MATRIX *
            QUADRATIC_COEFFICIENTS[Derivative] * BLENDING_MATRIX.transpose();

      if (J) {
        for (int i = 0; i < N; i++) {
          J->d_int_d_knot[i] = scaling * tmp.col(i);
        }
      }

      res = _Scalar(0.5) * scaling * (tmp.array() * knots_matrix.array()).sum();
    }

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  /// @brief Vector of derivatives of time polynomial.
  ///
  /// Computes a derivative of \f$ \begin{bmatrix}1 & t & t^2 & \dots &
  /// t^{N-1}\end{bmatrix} \f$ with repect to time. For example, the first
  /// derivative would be \f$ \begin{bmatrix}0 & 1 & 2 t & \dots & (N-1)
  /// t^{N-2}\end{bmatrix} \f$.
  /// @param Derivative derivative to evaluate
  /// @param[out] res_const vector to store the result
  /// @param[in] t
  template <int Derivative, class Derived>
  static void baseCoeffsWithTime(const Eigen::MatrixBase<Derived>& res_const,
                                 _Scalar t) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, N);
    Eigen::MatrixBase<Derived>& res =
        const_cast<Eigen::MatrixBase<Derived>&>(res_const);

    res.setZero();

    if (Derivative < N) {
      res[Derivative] = BASE_COEFFICIENTS(Derivative, Derivative);

      _Scalar ti = t;
      for (int j = Derivative + 1; j < N; j++) {
        res[j] = BASE_COEFFICIENTS(Derivative, j) * ti;
        ti = ti * t;
      }
    }
  }

  template <int, int, typename>
  friend class RdSpline;

  static const MatN
      BLENDING_MATRIX;  ///< Blending matrix. See \ref computeBlendingMatrix.

  static const MatN INV_BLENDING_MATRIX;  ///< Inverse blending matrix.

  static const std::array<MatN, _N>
      QUADRATIC_COEFFICIENTS;  ///< Matrices used to compute integral of the
                               ///< squared time derivatives

  static const MatN BASE_COEFFICIENTS;  ///< Base coefficients matrix.
                                        ///< See \ref computeBaseCoefficients.

  std::array<VecD, _N> knots_;          ///< Knots
  int64_t dt_ns_{0};                    ///< Knot interval in nanoseconds
  int64_t start_t_ns_{0};               ///< Start time in nanoseconds
  std::array<_Scalar, _N> pow_inv_dt_;  ///< Array with inverse powers of dt
};

template <int _DIM, int _N, typename _Scalar>
const typename RdBezier<_DIM, _N, _Scalar>::MatN
    RdBezier<_DIM, _N, _Scalar>::BASE_COEFFICIENTS =
        computeBaseCoefficients<_N, _Scalar>();

template <int _DIM, int _N, typename _Scalar>
const typename RdBezier<_DIM, _N, _Scalar>::MatN
    RdBezier<_DIM, _N, _Scalar>::BLENDING_MATRIX =
        computeBlendingMatrixBezier<_N, _Scalar>();

template <int _DIM, int _N, typename _Scalar>
const std::array<typename RdBezier<_DIM, _N, _Scalar>::MatN, _N>
    RdBezier<_DIM, _N, _Scalar>::QUADRATIC_COEFFICIENTS =
        computeQuadraticCoefficients<_N, _Scalar>();

template <int _DIM, int _N, typename _Scalar>
const typename RdBezier<_DIM, _N, _Scalar>::MatN
    RdBezier<_DIM, _N, _Scalar>::INV_BLENDING_MATRIX =
        computeBlendingMatrixBezier<_N, _Scalar>().inverse();

}  // namespace basalt

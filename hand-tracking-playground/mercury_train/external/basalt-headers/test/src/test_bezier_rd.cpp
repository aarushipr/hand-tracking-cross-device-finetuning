/**
BSD 3-Clause License

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
*/

#include <basalt/spline/spline_common.h>
#include <iostream>
#include "Eigen/Core"
#include "basalt/spline/rd_bezier.h"
#include "basalt/spline/rd_spline.h"
#include "gtest/gtest.h"
#include "test_utils.h"

template <int DIM, int N, int DERIV>
void testEvaluate(const basalt::RdBezier<DIM, N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::RdBezier<DIM, N>::VecD;
  using MatrixD = typename basalt::RdBezier<DIM, N>::MatD;

  typename basalt::RdBezier<DIM, N>::JacobianStruct J_spline;

  spline.template evaluate<DERIV>(t_ns, &J_spline);

  VectorD x0;
  x0.setZero();

  for (size_t i = 0; i < N; i++) {
    std::stringstream ss;

    ss << "d_val_d_knot" << i << " time " << t_ns;

    MatrixD J_a;
    J_a.setZero();

    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a.diagonal().setConstant(J_spline.d_val_d_knot[i - J_spline.start_idx]);
    }

    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::RdBezier<DIM, N> spline1 = spline;
          spline1.getKnot(i) += x;

          return spline1.template evaluate<DERIV>(t_ns);
        },
        x0);
  }
}

template <int DIM, int N, int DERIV>
void testTimeDeriv(const basalt::RdBezier<DIM, N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::RdBezier<DIM, N>::VecD;

  VectorD d_val_d_t = spline.template evaluate<DERIV + 1>(t_ns);

  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  test_jacobian(
      "d_val_d_t", d_val_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;
        return spline.template evaluate<DERIV>(t_ns + inc);
      },
      x0);
}

template <int DIM, int N, int DERIV>
void testEquality(const basalt::RdBezier<DIM, N> &bezier,
                  const basalt::RdSpline<DIM, N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::RdBezier<DIM, N>::VecD;

  VectorD res1 = bezier.template evaluate<DERIV>(t_ns);
  VectorD res2 = spline.template evaluate<DERIV>(t_ns);

  EXPECT_TRUE(res1.isApprox(res2, 1e-8))
      << "res1 " << res1.transpose() << " res2 " << res2.transpose()
      << std::endl;
}

template <int DIM, int N, int DERIV>
void testEvaluateSplineTransform() {
  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N, false);

  for (int i = 0; i < 2 * N; i++) {
    basalt::RdBezier<DIM, N> bezier = spline.getSegmentBezierCurve(i);

    for (int64_t t_ns = bezier.minTimeNs(); t_ns < bezier.maxTimeNs();
         t_ns += 1e8) {
      testEquality<DIM, N, DERIV>(bezier, spline, t_ns);
    }
  }
}

template <int DIM, int N, int DERIV>
void testEvaluateIntegral(const basalt::RdBezier<DIM, N> &spline) {
  using VectorD = typename basalt::RdSpline<DIM, N>::VecD;

  typename basalt::RdBezier<DIM, N>::IntegralJacobianStruct J_spline;

  double analytic_integral_squared =
      spline.template evaluateIntegralSquared<DERIV>(&J_spline);

  double numeric_integral_squared = 0;
  int64_t dt_ns = 1e6;
  for (int64_t t_ns = spline.minTimeNs() + dt_ns / 2; t_ns < spline.maxTimeNs();
       t_ns += dt_ns) {
    typename basalt::RdBezier<DIM, N>::VecD x =
        spline.template evaluate<DERIV>(t_ns);
    numeric_integral_squared += x.squaredNorm() * dt_ns * 1e-9;
  }

  EXPECT_NEAR(analytic_integral_squared, numeric_integral_squared, 1e-4)
      << "analytic_integral_squared " << analytic_integral_squared
      << " numeric_integral_squared " << numeric_integral_squared << std::endl;

  for (size_t i = 0; i < N; i++) {
    std::stringstream ss;

    ss << "d_int_d_knot" << i;

    VectorD x0;
    x0.setZero();

    test_jacobian(
        ss.str(), J_spline.d_int_d_knot[i].transpose(),
        [&](const VectorD &x) {
          UNUSED(x);
          basalt::RdBezier<DIM, N> spline1 = spline;
          spline1.getKnot(i) += x;

          Eigen::Matrix<double, 1, 1> res;
          res[0] = spline1.template evaluateIntegralSquared<DERIV>();

          return res;
        },
        x0);
  }
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredPos4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 0>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredVel4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 1>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredAcc4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 2>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredJerk4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 3>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredPos5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 0>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredVel5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 1>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredAcc5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 2>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredJerk5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 3>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredPos6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 0>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredVel6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 1>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredAcc6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 2>(spline);
}

TEST(BezierTest, UBBezierEvaluateIntegralSquaredJerk6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(3e9));
  spline.genRandomTrajectory();

  testEvaluateIntegral<DIM, N, 3>(spline);
}

TEST(BezierTest, UBBezierBsplineKnotTransforme4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;
  testEvaluateSplineTransform<DIM, N, 0>();
}

TEST(BezierTest, UBBezierBsplineKnotTransforme5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;
  testEvaluateSplineTransform<DIM, N, 0>();
}

TEST(BezierTest, UBBezierBsplineKnotTransforme6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;
  testEvaluateSplineTransform<DIM, N, 0>();
}

TEST(BezierTest, UBBezierEvaluateKnots4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierEvaluateKnots5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierEvaluateKnots6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityKnots4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 1>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityKnots5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 1>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityKnots6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 1>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierAccelKnots4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 2>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierAccelKnots5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 2>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierAccelKnots6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 2>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierEvaluateTimeDeriv4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierEvaluateTimeDeriv5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierEvaluateTimeDeriv6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityTimeDeriv4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 1>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityTimeDeriv5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 1>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityTimeDeriv6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 1>(spline, t_ns);
  }
}

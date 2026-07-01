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

#include <basalt/spline/so3_spline.h>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.h"

template <int N>
void testEvaluateSo3(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<N>::Vec3;
  using MatrixD = typename basalt::So3Spline<N>::Mat3;
  using SO3 = typename basalt::So3Spline<N>::SO3;

  typename basalt::So3Spline<N>::JacobianStruct J_spline;

  SO3 res = spline.evaluate(t_ns, &J_spline);

  VectorD x0;
  x0.setZero();

  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;

    ss << "d_val_d_knot" << i << " time " << t_ns;

    MatrixD J_a;
    J_a.setZero();

    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a = J_spline.d_val_d_knot[i - J_spline.start_idx];
    }

    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::So3Spline<N> spline1 = spline;
          spline1.getKnot(i) = SO3::exp(x) * spline.getKnot(i);

          SO3 res1 = spline1.evaluate(t_ns);

          return (res1 * res.inverse()).log();
        },
        x0);
  }
}

template <int N>
void testVelSo3(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<N>::Vec3;
  using SO3 = typename basalt::So3Spline<N>::SO3;

  SO3 res = spline.evaluate(t_ns);

  VectorD d_res_d_t = spline.velocityBody(t_ns);

  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  test_jacobian(
      "d_val_d_t", d_res_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;
        return (res.inverse() * spline.evaluate(t_ns + inc)).log();
      },
      x0);
}

template <int N>
void testAccelSo3(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<5>::Vec3;

  VectorD vel1;
  VectorD d_res_d_t = spline.accelerationBody(t_ns, &vel1);

  VectorD vel2 = spline.velocityBody(t_ns);
  EXPECT_TRUE(vel1.isApprox(vel2));

  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  test_jacobian(
      "d_val_d_t", d_res_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;
        return spline.velocityBody(t_ns + inc);
      },
      x0);
}

template <int N>
void testJerkSo3(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<5>::Vec3;

  VectorD vel1;
  VectorD accel1;
  VectorD d_res_d_t = spline.jerkBody(t_ns, &vel1, &accel1);

  VectorD vel2 = spline.velocityBody(t_ns);
  EXPECT_TRUE(vel1.isApprox(vel2));

  VectorD accel2 = spline.accelerationBody(t_ns);
  EXPECT_TRUE(accel1.isApprox(accel2));

  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  test_jacobian(
      "d_val_d_t", d_res_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;
        return spline.accelerationBody(t_ns + inc);
      },
      x0);
}

template <int N>
void testEvaluateSo3Vel(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<5>::Vec3;
  using MatrixD = typename basalt::So3Spline<5>::Mat3;
  using SO3 = typename basalt::So3Spline<5>::SO3;

  typename basalt::So3Spline<N>::JacobianStruct J_spline;

  VectorD res = spline.velocityBody(t_ns, &J_spline);
  VectorD res_ref = spline.velocityBody(t_ns);

  ASSERT_TRUE(res_ref.isApprox(res)) << "res and res_ref are not the same";

  VectorD x0;
  x0.setZero();

  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;

    ss << "d_vel_d_knot" << i << " time " << t_ns;

    MatrixD J_a;
    J_a.setZero();

    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a = J_spline.d_val_d_knot[i - J_spline.start_idx];
    }

    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::So3Spline<N> spline1 = spline;
          spline1.getKnot(i) = SO3::exp(x) * spline.getKnot(i);

          return spline1.velocityBody(t_ns);
        },
        x0);
  }
}

template <int N>
void testEvaluateSo3Accel(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<N>::Vec3;
  using MatrixD = typename basalt::So3Spline<N>::Mat3;
  using SO3 = typename basalt::So3Spline<N>::SO3;

  typename basalt::So3Spline<N>::JacobianStruct J_accel;
  typename basalt::So3Spline<N>::JacobianStruct J_vel;

  VectorD vel;
  VectorD vel_ref;

  VectorD res = spline.accelerationBody(t_ns, &J_accel, &vel, &J_vel);
  VectorD res_ref = spline.accelerationBody(t_ns, &vel_ref);

  ASSERT_TRUE(vel_ref.isApprox(vel)) << "vel and vel_ref are not the same";
  ASSERT_TRUE(res_ref.isApprox(res)) << "res and res_ref are not the same";

  VectorD x0;
  x0.setZero();

  // Test velocity Jacobian
  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;

    ss << "d_vel_d_knot" << i << " time " << t_ns;

    MatrixD J_a;
    J_a.setZero();

    if (i >= J_vel.start_idx && i < J_vel.start_idx + N) {
      J_a = J_vel.d_val_d_knot[i - J_vel.start_idx];
    }

    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::So3Spline<N> spline1 = spline;
          spline1.getKnot(i) = SO3::exp(x) * spline.getKnot(i);

          return spline1.velocityBody(t_ns);
        },
        x0);
  }

  // Test acceleration Jacobian
  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;

    ss << "d_accel_d_knot" << i << " time " << t_ns;

    MatrixD J_a;
    J_a.setZero();

    if (i >= J_accel.start_idx && i < J_accel.start_idx + N) {
      J_a = J_accel.d_val_d_knot[i - J_accel.start_idx];
    }

    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::So3Spline<N> spline1 = spline;
          spline1.getKnot(i) = SO3::exp(x) * spline.getKnot(i);

          return spline1.accelerationBody(t_ns);
        },
        x0);
  }
}

TEST(SplineSO3Test, SO3CUBSplineEvaluateKnots4) {
  static constexpr int N = 4;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineEvaluateKnots5) {
  static constexpr int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineEvaluateKnots6) {
  static constexpr int N = 6;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineVelocity4) {
  static constexpr int N = 4;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testVelSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineVelocity5) {
  static constexpr int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testVelSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineVelocity6) {
  static constexpr int N = 6;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testVelSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineAcceleration4) {
  static constexpr int N = 4;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testAccelSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineAcceleration5) {
  static constexpr int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testAccelSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineAcceleration6) {
  static constexpr int N = 6;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testAccelSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineJerk5) {
  static constexpr int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testJerkSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineJerk6) {
  static constexpr int N = 6;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testJerkSo3(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineVelocityKnots4) {
  static constexpr int N = 4;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Vel(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineVelocityKnots5) {
  static constexpr int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Vel(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineVelocityKnots6) {
  static constexpr int N = 6;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Vel(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineAccelerationKnots4) {
  static constexpr int N = 4;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Accel(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineAccelerationKnots5) {
  static constexpr int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Accel(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineAccelerationKnots6) {
  static constexpr int N = 6;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Accel(spline, t_ns);
  }
}

TEST(SplineSO3Test, SO3CUBSplineBounds) {
  static constexpr int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // std::cerr << "spline.maxTimeNs() " << spline.maxTimeNs() << std::endl;

  spline.evaluate(spline.maxTimeNs());
  // std::cerr << "res1\n" << res1.matrix() << std::endl;
  spline.evaluate(spline.minTimeNs());
  // std::cerr << "res2\n" << res2.matrix() << std::endl;

  // Sophus::SO3d res3 = spline.evaluate(spline.maxTimeNs() + 1);
  // std::cerr << "res3\n" << res1.matrix() << std::endl;
  // Sophus::SO3d res4 = spline.evaluate(spline.minTimeNs() - 1);
  // std::cerr << "res4\n" << res2.matrix() << std::endl;
}

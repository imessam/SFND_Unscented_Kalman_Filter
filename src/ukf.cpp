#include "ukf.h"
#include "Eigen/Dense"
#include "iostream"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{

  std::cout << "Initializing..." << std::endl;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.fill(0);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_.fill(0);
  P_(0,0) = 1/5;
  P_(1,1) = 1/5;
  P_(2,2) = 1/5;
  P_(3,3) = 1/5;
  P_(4,4) = 1/5;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 9.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2*M_PI;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  time_us_ = 0;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  int n_a = 2 * n_aug_ + 1;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0);

  // Initialize weights.
  weights_ = VectorXd(n_a);
  weights_.fill(0);
  // set vector for weights
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  double weight = 0.5 / (lambda_ + n_aug_);
  weights_(0) = weight_0;

  for (int i = 1; i < n_a; ++i)
  {
    weights_(i) = weight;
  }

  std::cout << "Initialized from constructor ... " << std::endl;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{

  std::cout << "Processing measurements .. " << std::endl;

  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  
  if (!is_initialized_)
  {

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
      return;

    std::cout << "Kalman Filter Initialization " << std::endl;

    // set the state with the initial location and zero velocity
    x_(0) = meas_package.raw_measurements_(0);
    x_(1) = meas_package.raw_measurements_(1);

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    std::cout << "Kalman Filter Initialized.... " << std::endl;

    return;
  }

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_)
    return;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_)
    return;

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  std::cout << time_us_ << " : " << meas_package.timestamp_ << std::endl;
  time_us_ = meas_package.timestamp_;

  // 3. Call the Kalman Filter predict() function
  Prediction(dt);
  // 4. Call the Kalman Filter update() function
  //      with the most recent raw measurements_
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    UpdateLidar(meas_package);
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    UpdateRadar(meas_package);
}

void UKF::Prediction(double delta_t)
{

  std::cout << "Predicting ..." << delta_t << std::endl;
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
  GenerateSigmaPoints(&Xsig);

  // print result
  std::cout << "Xsig = " << std::endl
            << Xsig << std::endl;

  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  AugmentedSigmaPoints(Xsig, &Xsig_aug);

  // print result
  std::cout << "Xsig_aug = " << std::endl
            << Xsig_aug << std::endl;

  SigmaPointPrediction(Xsig_aug, delta_t, &Xsig_pred_);

  // print result
  std::cout << "Xsig_pred = " << std::endl
            << Xsig_pred_ << std::endl;

  PredictMeanAndCovariance(&x_, &P_);

  // print result
  std::cout << "x_pred = " << std::endl
            << x_ << std::endl;
  std::cout << "P_pred = " << std::endl
            << P_ << std::endl;
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  std::cout << "Updating Lidar ...." << std::endl;

  MatrixXd Zsig_out = MatrixXd(2, 2 * n_aug_ + 1);
  VectorXd z_out = VectorXd(2);
  MatrixXd S_out = MatrixXd(2, 2);
  PredictLidarMeasurement(&Zsig_out, &z_out, &S_out);

  // print result
  std::cout << "z_out = " << std::endl
            << z_out << std::endl;
  std::cout << "S_out = " << std::endl
            << S_out << std::endl;

  UpdateLidarState(Zsig_out, z_out, S_out, meas_package.raw_measurements_, &x_, &P_);

  // print result
  std::cout << "x_out = " << std::endl
            << x_ << std::endl;
  std::cout << "P_out = " << std::endl
            << P_ << std::endl;

}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  std::cout << "Updating Radar ...." << std::endl;

  MatrixXd Zsig_out = MatrixXd(3, 2 * n_aug_ + 1);
  VectorXd z_out = VectorXd(3);
  MatrixXd S_out = MatrixXd(3, 3);
  PredictRadarMeasurement(&Zsig_out, &z_out, &S_out);

  // print result
  std::cout << "z_out = " << std::endl
            << z_out << std::endl;
  std::cout << "S_out = " << std::endl
            << S_out << std::endl;

  UpdateRadarState(Zsig_out, z_out, S_out, meas_package.raw_measurements_, &x_, &P_);

  // print result
  std::cout << "x_out = " << std::endl
            << x_ << std::endl;
  std::cout << "P_out = " << std::endl
            << P_ << std::endl;
}

void UKF::GenerateSigmaPoints(MatrixXd *Xsig_out)
{

  std::cout << "Generating sigma points ...." << std::endl;

  // create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
  Xsig.fill(0);

  // calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  /**
   * Student part begin
   */
  Xsig.col(0) = x_;
  for (int i = 0; i < n_x_; ++i)
  {
    Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }

  /**
   * Student part end
   */

  // write result
  *Xsig_out = Xsig;
}

void UKF::AugmentedSigmaPoints(MatrixXd &Xsig, MatrixXd *Xsig_out)
{

  std::cout << "Augmenting sigma points ...." << std::endl;

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);
  x_aug.fill(0);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  P_aug.fill(0);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0);

  /**
   * Student part begin
   */

  // create augmented mean state
  x_aug.head(5) = x_;

  // create augmented covariance
  MatrixXd Q(2, 2);
  Q << pow(std_a_, 2), 0,
      0, pow(std_yawdd_, 2);
  P_aug.topLeftCorner(P_.rows(), P_.cols()) = P_;
  P_aug.bottomRightCorner(Q.rows(), Q.cols()) = Q;

  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  /**
   * Student part end
   */

  // write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd &Xsig_aug, double delta_t, MatrixXd *Xsig_out)
{

  std::cout << "Predicting Sigma points ...." << std::endl;

  // create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred.fill(0);

  /**
   * Student part begin
   */

  VectorXd x, process(n_x_), noise(n_x_);
  double v, yaw, yaw_dot, acc_noise, yaw_acc_noise;

  for (int col = 0; col < Xsig_aug.cols(); col++)
  {

    x = Xsig_aug.col(col);
    v = x(2);
    yaw = x(3);
    yaw_dot = x(4);
    acc_noise = x(5);
    yaw_acc_noise = x(6);

    process.fill(0);
    process(0) =
        yaw_dot == 0 ? v * cos(yaw) * delta_t : (v / yaw_dot) * (sin(yaw + (yaw_dot * delta_t)) - sin(yaw));
    process(1) =
        yaw_dot == 0 ? v * sin(yaw) * delta_t : (v / yaw_dot) * (-cos(yaw + (yaw_dot * delta_t)) + cos(yaw));
    process(3) = yaw_dot == 0 ? 0 : yaw_dot * delta_t;

    noise.fill(0);
    noise(0) = 0.5 * pow(delta_t, 2) * cos(yaw) * acc_noise;
    noise(1) = 0.5 * pow(delta_t, 2) * sin(yaw) * acc_noise;
    noise(2) = delta_t * acc_noise;
    noise(3) = 0.5 * pow(delta_t, 2) * yaw_acc_noise;
    noise(4) = delta_t * yaw_acc_noise;

    Xsig_pred.col(col) = Xsig_aug.col(col).head(n_x_) + process + noise;
  }
  /**
   * Student part end
   */

  // write result
  *Xsig_out = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(VectorXd *x_out, MatrixXd *P_out)
{
  std::cout << "Predicting mean and covariance ...." << std::endl;

  // create vector for predicted state
  VectorXd x = VectorXd(n_x_);
  x.fill(0);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0);

  int n_a = 2 * n_aug_ + 1;

  /**
   * Student part begin
   */

  // predict state mean
  for (int col = 0; col < n_a; col++)
    x = x + (weights_(col) * Xsig_pred_.col(col));

  // predict state covariance matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  { // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose();
  }

  /**
   * Student part end
   */
  std::cout << "finishes" << std::endl;
  // write result
  *x_out = x;
  *P_out = P;
}

void UKF::PredictLidarMeasurement(MatrixXd *Zsig_out, VectorXd *z_out, MatrixXd *S_out)
{

  std::cout << "Predict Lidar measurement ...." << std::endl;

  int n_z = 2;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0);

  /**
   * Student part begin
   */

  MatrixXd R(n_z, n_z);
  R.fill(0);

  R(0, 0) = pow(std_laspx_, 2);
  R(1, 1) = pow(std_laspy_, 2);

  int n_a = 2 * n_aug_ + 1;
  double px, py, v, yaw, yaw_dot;
  VectorXd x;

  // transform sigma points into measurement space
  for (int col = 0; col < n_a; col++)
  {

    x = Xsig_pred_.col(col);

    px = x(0);
    py = x(1);
    v = x(2);
    yaw = x(3);
    yaw_dot = x(4);


    Zsig.col(col)(0) = px;
    Zsig.col(col)(1) = py;
  }

  // calculate mean predicted measurement
  for (int col = 0; col < n_a; col++)
    z_pred = z_pred + (weights_(col) * Zsig.col(col));

  // calculate innovation covariance matrix S
  for (int i = 0; i < n_a; ++i)
  { // iterate over sigma points
    // state difference
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S += R;

  /**
   * Student part end
   */

  // write result
  *Zsig_out = Zsig;
  *z_out = z_pred;
  *S_out = S;
}

void UKF::PredictRadarMeasurement(MatrixXd *Zsig_out, VectorXd *z_out, MatrixXd *S_out)
{

  std::cout << "Predict Radar measurement ...." << std::endl;

  int n_z = 3;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0);

  /**
   * Student part begin
   */

  MatrixXd R(n_z, n_z);
  R.fill(0);

  R(0, 0) = pow(std_radr_, 2);
  R(1, 1) = pow(std_radphi_, 2);
  R(2, 2) = pow(std_radrd_, 2);

  int n_a = 2 * n_aug_ + 1;
  double px, py, v, yaw, yaw_dot;
  double rho, phi, rho_dot;
  VectorXd x;

  // transform sigma points into measurement space
  for (int col = 0; col < n_a; col++)
  {

    x = Xsig_pred_.col(col);

    px = x(0);
    py = x(1);
    v = x(2);
    yaw = x(3);
    yaw_dot = x(4);

    // while (yaw > M_PI)
    //   yaw -= 2. * M_PI;
    // while (yaw < -M_PI)
    //   yaw += 2. * M_PI;

    rho = sqrt(pow(px, 2) + pow(py, 2));
    phi = atan2(py , px);
    rho_dot = ((px * cos(yaw) * v) + (py * sin(yaw) * v)) / (sqrt(pow(px, 2) + pow(py, 2)));

    Zsig.col(col)(0) = rho;
    Zsig.col(col)(1) = phi;
    Zsig.col(col)(2) = rho_dot;
  }

  // calculate mean predicted measurement
  for (int col = 0; col < n_a; col++)
    z_pred = z_pred + (weights_(col) * Zsig.col(col));

  // calculate innovation covariance matrix S
  for (int i = 0; i < n_a; ++i)
  { // iterate over sigma points
    // state difference
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S += R;

  /**
   * Student part end
   */

  // write result
  *Zsig_out = Zsig;
  *z_out = z_pred;
  *S_out = S;
}

void UKF::UpdateLidarState(MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S, VectorXd &z, VectorXd *x_out, MatrixXd *P_out)
{

  std::cout << "Updating Lidar State ...." << std::endl;

  int n_z = 2;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);

  /**
   * Student part begin
   */

  int n_a = 2 * n_aug_ + 1;

  // calculate cross correlation matrix
  for (int i = 0; i < n_a; ++i)
  { // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_, z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix

  // residual
  VectorXd z_diff = z - z_pred;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  /**
   * Student part end
   */

  // write result
  *x_out = x_;
  *P_out = P_;
}

void UKF::UpdateRadarState(MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S, VectorXd &z, VectorXd *x_out, MatrixXd *P_out)
{

  std::cout << "Updating Radar State ...." << std::endl;

  int n_z = 3;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);

  /**
   * Student part begin
   */

  int n_a = 2 * n_aug_ + 1;

  // calculate cross correlation matrix
  for (int i = 0; i < n_a; ++i)
  { // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_, z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  /**
   * Student part end
   */

  // write result
  *x_out = x_;
  *P_out = P_;
}

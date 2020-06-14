/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Reference: Dhanoop Karunakaran's article on medium.com
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;
using std::discrete_distribution;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
    default_random_engine gen;  // Random engine from std lib
    double std_x, std_y, std_theta;  // Standard deviations for x, y, and theta

    num_particles = 100;  // Set the number of particles

      if(is_initialized) { // Normal Error handling to avoid re-initialization
        return;
    }

    //This is added to improve redability of the code
    std_x = std[0]; // Standard deviation of x [m]
    std_y = std[1]; // Standard deviation of y [m]
    std_theta = std[2]; // standard deviation of yaw [rad]

    // Normal Distributions
    normal_distribution<double> dist_x(x, std_x); // Initial x position from GPS as mean
    normal_distribution<double> dist_y(y, std_y); // Initial y position from GPS as mean
    normal_distribution<double> dist_theta(theta, std_theta); //initial heading angle from GPS as mean

    //Generate 'num_particles' particles
    for(int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }
    is_initialized = true; // To avoid re-initialization

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    default_random_engine gen;
    double std_x, std_y, std_theta;

    //This is added to improve redability of the code
    std_x = std_pos[0];     // standard deviation of x [m]
    std_y = std_pos[1];     // standard deviation of y [m]
    std_theta = std_pos[2]; // standard deviation of yaw [rad]]

    //Normal distributions
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    // Based on the formulae provided in sessions, Predict x, y and theta.
    // Using standard trigonometric ratios sin(θ) and cos(θ)
    for(int i = 0; i < num_particles; i++) {
        if(fabs(yaw_rate) < 0.00001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);

            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else{
            particles[i].x += velocity / yaw_rate * \
            (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));

            particles[i].y += velocity / yaw_rate * \
            (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));

            particles[i].theta += yaw_rate * delta_t;
        }

        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
	for(unsigned int i = 0; i < observations.size(); i++) {

        // These variable are added to improve redability, compiler will optimize them away
		unsigned int sizeOfObservations = observations.size();
		unsigned int sizeOfPredications = predicted.size();

		for(unsigned int i = 0; i < sizeOfObservations; i++) {

			double currentMinimum = 0;
			int id = -1;

			for(unsigned j = 0; j < sizeOfPredications; j++ ) {

				double xDist = observations[i].x - predicted[j].x;
				double yDist = observations[i].y - predicted[j].y;
				double distance = (xDist * xDist) + (yDist * yDist);

				if(currentMinimum == 0) {
                    currentMinimum = distance; // First Value
				}
				else if(distance < currentMinimum) {
                    // If the distance is lower than current min then store it as min
					currentMinimum = distance;
					id = predicted[j].id;
				}
				observations[i].id = id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
	for(int i = 0; i < num_particles; i++) {

		vector<LandmarkObs> predictions;
		unsigned int noOfLandmarks = map_landmarks.landmark_list.size();

        double paricle_x = particles[i].x;
		double paricle_y = particles[i].y;
		double paricle_theta = particles[i].theta;

		for(unsigned int j = 0; j < noOfLandmarks; j++) {

			float lm_x = map_landmarks.landmark_list[j].x_f;
			float lm_y = map_landmarks.landmark_list[j].y_f;
			int lm_id = map_landmarks.landmark_list[j].id_i;

			if(pow((lm_x - paricle_x), 2) + pow((lm_x - paricle_x), 2) <= sensor_range*sensor_range) {
				predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
			}
		}

		vector<LandmarkObs> transformed;
		for(unsigned int j = 0; j < observations.size(); j++) {
			double t_x = cos(paricle_theta)*observations[j].x - sin(paricle_theta)*observations[j].y + paricle_x;
			double t_y = sin(paricle_theta)*observations[j].x + cos(paricle_theta)*observations[j].y + paricle_y;
			transformed.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
		}

		dataAssociation(predictions, transformed);
		particles[i].weight = 1.0;
		for(unsigned int j = 0; j < transformed.size(); j++) {
			double o_x, o_y, pr_x, pr_y;
			o_x = transformed[j].x;
			o_y = transformed[j].y;
			int asso_prediction = transformed[j].id;

			for(unsigned int k = 0; k < predictions.size(); k++) {
				if(predictions[k].id == asso_prediction) {
					pr_x = predictions[k].x;
					pr_y = predictions[k].y;
				}
			}

			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(pr_x-o_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-o_y,2)/(2*pow(s_y, 2))) ) );

            if (obs_w == 0) {
                particles[i].weight *= 0.00001;
            } else {
                particles[i].weight *= obs_w;
            }
		}
	}
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  default_random_engine gen;
  vector<double> weights;

  for(unsigned int i = 0; i < particles.size(); i++) {
    weights.push_back(particles[i].weight);
    }

  std::discrete_distribution<> dd(weights.begin(), weights.end());

  vector<Particle> new_particles;

  for(int i = 0; i < num_particles; i++){
    int index = dd(gen);
    new_particles.push_back(particles[index]);
    }

  particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

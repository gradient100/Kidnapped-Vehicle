/*
 * particle_filter.cpp
 *
 *  Created on: May 12, 2018
 *      Author: Vinh Nghiem
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	time_step = 0.1;

  	default_random_engine generator;
  	normal_distribution <double> dist_x(x, std[0]);
  	normal_distribution <double> dist_y(y, std[1]);
  	normal_distribution <double> dist_theta(theta, std[2]);
  	
  	//cout << "Time Step : " << time_step;

	for (int i = 0; i < num_particles; i++)
	{
		Particle p;
		p.id = i;
		p.x = dist_x(generator);
		p.y = dist_y(generator);
		p.theta = dist_theta(generator);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(p.weight);

	}

	is_initialized = true;

	cout << "Initialization " <<endl;
	print();

}

void ParticleFilter::print(){
	for (int i=0; i< particles.size() ; i++)
	{	
		Particle p = particles[i];
		cout << "Particle #"<< i << ": " << "\t" 
			 << "id: "<< p.id << "\t"
			 << "x: "<<p.x<<"\t"
			 << "y: "<< p.y << "\t"
			 << "theta: "<<p.theta<<"\t"
			 << "weight: "<< p.weight  << endl;
	}			
}


void ParticleFilter::prediction(double dt, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	time_step += 0.1;

	default_random_engine generator;
  	normal_distribution <double> dist_x(0, std_pos[0]);
  	normal_distribution <double> dist_y(0, std_pos[1]);
  	normal_distribution <double> dist_theta(0, std_pos[2]);
  	
	for (int i = 0; i < num_particles; i++)
	{
		Particle & p = particles[i];
		if (yaw_rate==0)
			yaw_rate = 0.000001;
		p.x += velocity / yaw_rate * ( sin(p.theta + yaw_rate * dt) - sin(p.theta) ) + dist_x(generator) ;
		p.y += velocity / yaw_rate * ( cos(p.theta)  - cos(p.theta+yaw_rate*dt) ) + dist_y(generator) ;
		p.theta += yaw_rate*dt + dist_theta(generator);
	}

	// cout << endl << "Time Step : " << time_step << endl;
	// cout << "Prediction : " << endl;
	// print();

}

void ParticleFilter::dataAssociation(const Map &predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i=0; i < observations.size(); i++)
	{
		LandmarkObs & ob = observations[i];
	
		//double minDist =10000000; // 10 million
		double minDist = dist(ob.x, ob.y, predicted.landmark_list[0].x_f, predicted.landmark_list[0].y_f);
		for (int j=0; j < predicted.landmark_list.size(); j++)
		{
			Map::single_landmark_s lm = predicted.landmark_list[j];
			double distance = dist(ob.x, ob.y, lm.x_f, lm.y_f);
			if (distance <= minDist)
			{
				minDist = distance;
				ob.id = lm.id_i; 
				
			}

		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();
	for (int i=0; i < particles.size(); i++)
	{
		Particle &p = particles[i];
		double weight = 1.0; 
		vector<LandmarkObs> mobservations; // obs in map coords
		for (int j=0; j < observations.size(); j++)
		{	
			
			LandmarkObs ob = observations[j];
			
			LandmarkObs mobservation;
			mobservation.x = cos(p.theta)*ob.x - sin(p.theta)*ob.y + p.x;
			mobservation.y = sin(p.theta)*ob.x + cos(p.theta)*ob.y + p.y;
			mobservations.push_back(mobservation);
		}
		dataAssociation(map_landmarks,mobservations);
		for (int j=0; j < mobservations.size(); j++)
		{	
			LandmarkObs mobservation = mobservations[j];
			//cout << "Ob id " << mobservation.id << endl;
			for (int k=0; k < map_landmarks.landmark_list.size(); k++)
			{
				Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
				//cout << "landmark id = " << landmark.id_i << endl;
				if (mobservation.id == landmark.id_i && dist(p.x, p.y, mobservation.x,mobservation.y)<=sensor_range)
				//if (mobservation.id == landmark.id_i)
				{
					//cout << "ID Matched! Particle # " <<i<< ", Obs #" << j << " and LM #" << k << ", landmark id = " << landmark.id_i <<endl;
					weight=weight*Gaussian2D(mobservation.x,mobservation.y,landmark.x_f,landmark.y_f, std_landmark[0], std_landmark[1]);
				}
			}
		}
		p.weight = weight;
		weights.push_back(weight);  //***** Normalize weights ?????
		// cout << endl;
	}
	// cout << "Update : " << endl;
	// print();

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine generator;
	discrete_distribution<int> dist (weights.begin(),weights.end());
	
	vector<Particle> new_particles;
	vector <double> new_weights;

	for (int i =0; i < num_particles; i++)
	{
		int weightedRand = dist(generator);
		new_particles.push_back(particles[weightedRand]);
		new_weights.push_back(weights[weightedRand]);
	}
	particles = new_particles;
	weights = new_weights;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

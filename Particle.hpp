#include <vector>

class Particle {
public:
	void init_location() = 0;
	void init_velocity() = 0;
	void init_personal_best() = 0;

	std::vector<Particle> filter_neighbors(std::vector<Particle> other_particles) = 0;

	void update_velocity() = 0;
	void translate() = 0;

protected:
	std::vector<double> location;
	std::vector<double> personal_best_location;
	double personal_best_value;
	std::vector<double> velocity;
};
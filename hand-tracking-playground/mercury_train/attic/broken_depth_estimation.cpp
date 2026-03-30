bool
quadratic_equation_solve(float a, float b, float c, float *x1, float *x2)
{
	float in_sqrt = (b * b) - 4 * a * c;
	// U_LOG_E("in_sqrt is %f. %f %f", in_sqrt, b*b, 4*a*c);
	if (in_sqrt > 0) {

		*x1 = (-b + sqrt(in_sqrt)) / (2 * a);
		*x2 = (-b - sqrt(in_sqrt)) / (2 * a);
		return true;
	} else if (in_sqrt == 0) {
		*x1 = (-b) / (2 * a);
		*x2 = (-b) / (2 * a);
		return true;
	}
	return false;
}


// This should do the same as sympy - it's hand-derived. Sympy is right, this is wrong.
float
broke(float angle, float wrist_extra_distance_meters, float hand_size)
{
	float a = 2;
	float b = 2 * wrist_extra_distance_meters;
	float c = -1 * (hand_size * hand_size) / (-angle + 1);

	float d1 = 0;
	float d2 = 0;

	bool success = quadratic_equation_solve(a, b, c, &d1, &d2);

	// U_LOG_E("Hand size was %5f, wed was %5f, angle was %f. Success was %d, Distances are %5f and %5f", hand_size,
	// wrist_extra_distance_meters, angle, success, d1, d2);
	U_LOG_E(" broke: Distances are %5f and %5f", hand_size, wrist_extra_distance_meters, angle, success, d1, d2);
	return d1;
}

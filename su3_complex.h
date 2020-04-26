#ifndef H_SU3_COMPLEX
#define H_SU3_COMPLEX

template <class T>
class su3_complex {
	public:
		T r, i;

		su3_complex(T rr, T ii) {
			this->r = rr;
			this->i = ii;
		}

		su3_complex() {
			this->r = 0.0;
			this->i = 0.0;
		}

		su3_complex operator*(T alpha) {

			this->r *= alpha;
			this->i *= alpha;
			return *this;
		}

		su3_complex operator=(const T& a) {

			this->r = a;
			this->i = 0.0;
			return *this;
		}

		su3_complex operator+(const su3_complex &b) {

			su3_complex c;
			c.r = r + b.r;
			c.i = i + b.i;

			return c;
		}

		su3_complex operator-(const su3_complex &b) {
			su3_complex c;
			c.r = r - b.r;
			c.i = i - b.i;

			return c;
		}

		su3_complex operator*(const su3_complex &b) {

			su3_complex c;
			c.r = r * b.r - i * b.i;
			c.i = r * b.i + i * b.r;

			return c;
		}

		su3_complex dagger_mult(const su3_complex &b) {

			su3_complex c;
			c.r = r * b.r + i * b.i;
			c.i = r * b.i - i * b.r;

			return c;
		}

		su3_complex mult(const su3_complex &b) {

			su3_complex c;
			c.r = r * b.r - i * b.i;
			c.i = r * b.i + i * b.r;

			return c;
		}

		su3_complex conj(){
			
			su3_complex z;
			z.r = r;
			z.i = -i;
			return z;
		}

};

#endif

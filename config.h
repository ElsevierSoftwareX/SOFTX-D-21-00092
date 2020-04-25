#define Nx 2
#define Ny 2
#define Nz 8
#define Tt 2

extern int Nxl;
extern int Nyl;
extern int Nzl;

extern int Nxl_buf;
extern int Nyl_buf;
extern int Nzl_buf;

extern int ExchangeX;
extern int ExchangeY;
extern int ExchangeZ;

extern int XNeighbourNext, XNeighbourPrevious;
extern int YNeighbourNext, YNeighbourPrevious;
extern int ZNeighbourNext, ZNeighbourPrevious;

extern int procx;
extern int procy;
extern int procz;

extern double *x;
extern double *p;

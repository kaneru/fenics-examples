#include <dolfin.h>
#include "Parabolic.h"
#include "Error.h"

using namespace dolfin;

class FExpression : public Expression {
public:
    FExpression(double& t) : Expression(), t(t) {
    }

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = -6 * exp(-x[0] - 2 * x[1] - t);
    }
private:
    double& t;
};

class DirichletValue : public Expression {
public:
    DirichletValue(double& t) : Expression(), t(t) {
    }

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = exp(-x[0] - 2 * x[1] - t);
    }
private:
    double& t;
};

class DirichletBoundary : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const {
        return on_boundary;
    }
};

class ExactSolution : public Expression {
public:
    ExactSolution(double& t) : Expression(), t(t) {
    }

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = exp(-x[0] - 2 * x[1] - t);
    }
private:
    double& t;
};

int main() {
    UnitSquare mesh(16, 16);
    double t = 0;
    double tau = 0.1;
    double T = 1;

    Parabolic::FunctionSpace V(mesh);
    Function u(V);
    FExpression f(t);
    ExactSolution exact(t);
    Function u0(V);
    Constant dt(tau);

    DirichletValue g(t);
    DirichletBoundary Gamma;
    DirichletBC bc(V, g, Gamma);

    Parabolic::BilinearForm a(V, V);
    a.dt = dt;
    Parabolic::LinearForm L(V);
    L.dt = dt; L.f = f; L.u0 = u0;

    Error::Functional error(mesh);
    error.u = u;
    error.exact = exact;

    Matrix A;
    assemble(A, a);
    Vector b;

    File file("Solution.pvd");

    while (t <= T) {
        t += tau;
        begin("Computing time %G", t);

        assemble(b, L);

        bc.apply(A, b);

        solve(A, *u.vector(), b);

        double error_norm = sqrt(assemble(error));
        info("Error norm = %G", error_norm);

        file << u;
        u0 = u;
        end();
    }
    
    plot(u); interactive();
    
    return 0;
}

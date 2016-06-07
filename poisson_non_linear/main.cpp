#include <dolfin.h>
#include "Error.h"
#include "Poisson.h"

const double m = 2;

using namespace dolfin;

class FExpression : public Expression {
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = 0;
    }
};

class DirichletValue : public Expression {
    void eval(Array<double>& values, const Array<double>& x) const {
        if (x[0] == 0)
            values[0] = 0;
        else if (x[0] == 1)
            values[0] = 1;
    }
};

class DirichletBoundary : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const {
        return on_boundary && (std::abs(x[0] - 0.0) < DOLFIN_EPS || std::abs(x[0] - 1.0) < DOLFIN_EPS);
    }
};

class ExactSolution : public Expression {
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = pow((pow(2, m + 1) - 1) * x[0] + 1, 1.0/(m + 1)) - 1;
    }
};

int main() {
    UnitSquare mesh(16, 16);

    Poisson::FunctionSpace V(mesh);

    DirichletValue g;
    DirichletBoundary Gamma;
    DirichletBC bc(V, g, Gamma);

    FExpression f;
    Function u(V);

    Poisson::LinearForm F(V);
    F.u = u; F.f = f;

    Poisson::JacobianForm J(V, V);
    J.u = u;

    solve(F == 0, u, bc, J);

    plot(u); interactive();

    ExactSolution exact;

    Error::Functional error(mesh);
    error.u = u;
    error.exact = exact;
    double error_norm = sqrt(assemble(error));

    info("Error norm = %G", error_norm);

    File file("Solution.pvd");
    file << u;

    return 0;
}

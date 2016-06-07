#include <dolfin.h>
#include "Poisson.h"
#include "Error.h"

using namespace dolfin;

class FExpression : public Expression {
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = -5 * exp(-x[0] - 2 * x[1]);
    }
};

class DirichletValue : public Expression {
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = exp(-x[0] - 2 * x[1]);
    }
};

class DirichletBoundary : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const {
        return on_boundary;
    }
};

class ExactSolution : public Expression {
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = exp(-x[0] - 2 * x[1]);
    }
};

int main() {
    UnitSquare mesh(32, 32);

    Poisson::FunctionSpace V(mesh);
    Poisson::BilinearForm a(V, V);
    Poisson::LinearForm L(V);

    FExpression f;
    L.f = f;

    DirichletValue g;
    DirichletBoundary Gamma;
    DirichletBC bc(V, g, Gamma);

    std::vector<const DirichletBC*> bcs;
    bcs.push_back(&bc);

    Function u(V);
    solve(a == L, u, bcs);

    // plot(u);
    // interactive();

    ExactSolution exact;
    plot(exact, mesh);
    interactive();

    File file("Solution.pvd");
    file << u;
    
    Error::Functional error(mesh);
    error.u = u;
    error.exact = exact;
    double error_norm = sqrt(assemble(error));

    info("Error norm = %G", error_norm);
    return 0;
}

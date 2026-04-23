// ============================================================================
// bridge.cpp — nanobind Python boundary for the HLFIR bridge.
// ============================================================================
// Thin layer: owns an MLIRContext and a ModuleOp; delegates extraction to
// extract_vars.cpp and extract_ast.cpp; delegates passes to the pass library
// via the MLIR pass pipeline parser.  Nothing here walks the IR.
// ============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

// MLIR core
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

// Flang FIR + HLFIR
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"

// Standard MLIR dialects needed by Flang's HLFIR output
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "bridge/extract_vars.h"
#include "bridge/extract_ast.h"
#include "passes/Passes.h"

#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace hlfir_bridge;

// ============================================================================
// HLFIRModule — Python-facing container for one parsed HLFIR module.
// ============================================================================

class HLFIRModule {
public:
    HLFIRModule() {
        registry_.insert<
            fir::FIROpsDialect, hlfir::hlfirDialect,
            mlir::arith::ArithDialect, mlir::func::FuncDialect,
            mlir::scf::SCFDialect, mlir::math::MathDialect,
            mlir::complex::ComplexDialect, mlir::DLTIDialect,
            mlir::cf::ControlFlowDialect, mlir::vector::VectorDialect,
            mlir::affine::AffineDialect, mlir::omp::OpenMPDialect,
            mlir::acc::OpenACCDialect, mlir::LLVM::LLVMDialect>();
        ctx_.appendDialectRegistry(registry_);
        ctx_.loadAllAvailableDialects();
    }

    bool parse(const std::string &t) {
        module_ = mlir::parseSourceString<mlir::ModuleOp>(
            llvm::StringRef(t), &ctx_);
        return static_cast<bool>(module_);
    }

    bool parse_file(const std::string &p) {
        module_ = mlir::parseSourceFile<mlir::ModuleOp>(
            llvm::StringRef(p), &ctx_);
        return static_cast<bool>(module_);
    }

    /// Run an mlir-opt-syntax pipeline.  Example:
    ///   run_passes("builtin.module(hlfir-propagate-shapes)")
    /// Every bridge pass is registered by registerAllBridgePasses() in
    /// NB_MODULE init, so pass names from passes/*.cpp are usable here.
    void run_passes(const std::string &pipeline) {
        if (!module_)
            throw std::runtime_error("run_passes: no module parsed");
        mlir::PassManager pm(&ctx_);
        if (mlir::failed(mlir::parsePassPipeline(pipeline, pm)))
            throw std::runtime_error("run_passes: bad pipeline: " + pipeline);
        if (mlir::failed(pm.run(*module_)))
            throw std::runtime_error("run_passes: pipeline failed");
    }

    /// Print the current IR as text (useful for debugging from Python).
    std::string dump() {
        if (!module_) return "";
        std::string s;
        llvm::raw_string_ostream os(s);
        module_->print(os);
        return s;
    }

    std::vector<VarInfo> get_variables() {
        if (!module_) return {};
        return extractVariables(*module_);
    }

    std::vector<ASTNode> get_ast() {
        if (!module_) return {};
        return extractAST(*module_);
    }

private:
    mlir::DialectRegistry registry_;
    mlir::MLIRContext ctx_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;
};

// ============================================================================
// nanobind bindings
// ============================================================================

NB_MODULE(hlfir_bridge, m) {
    m.doc() = "HLFIR -> Python bridge: parses Flang HLFIR, runs passes, "
              "exposes VarInfo list and recursive ASTNode tree.";

    // Register the bridge pass library exactly once per process.
    // Safe to call repeatedly; registerAllBridgePasses() is idempotent.
    registerAllBridgePasses();

    nb::class_<VarInfo>(m, "VarInfo")
        .def_ro("fortran_name",  &VarInfo::fortran_name)
        .def_ro("mangled_name",  &VarInfo::mangled_name)
        .def_ro("intent",        &VarInfo::intent)
        .def_ro("rank",          &VarInfo::rank)
        .def_ro("dtype",         &VarInfo::dtype)
        .def_ro("is_dynamic",    &VarInfo::is_dynamic)
        .def_ro("shape_symbols", &VarInfo::shape_symbols)
        .def_ro("lower_bounds",  &VarInfo::lower_bounds)
        .def_ro("role",          &VarInfo::role)
        .def("__repr__", [](const VarInfo &v) {
            std::string s = "<" + v.role + " '" + v.fortran_name + "'";
            if (v.rank > 0) {
                s += "(";
                for (size_t i = 0; i < v.shape_symbols.size(); ++i) {
                    if (i) s += ",";
                    if (i < v.lower_bounds.size() && v.lower_bounds[i] != "1")
                        s += v.lower_bounds[i] + ":";
                    s += v.shape_symbols[i];
                }
                s += ")";
            }
            s += " " + v.dtype;
            if (!v.intent.empty()) s += " intent(" + v.intent + ")";
            s += ">";
            return s;
        });

    nb::class_<AccessInfo>(m, "AccessInfo")
        .def_ro("array_name",  &AccessInfo::array_name)
        .def_ro("index_vars",  &AccessInfo::index_vars)
        .def_ro("index_exprs", &AccessInfo::index_exprs)
        .def_ro("is_read",     &AccessInfo::is_read)
        .def_ro("is_write",    &AccessInfo::is_write);

    nb::class_<ASTNode>(m, "ASTNode")
        .def_ro("kind",            &ASTNode::kind)
        .def_ro("loop_iter",       &ASTNode::loop_iter)
        .def_ro("loop_bound",      &ASTNode::loop_bound)
        .def_ro("loop_lower",      &ASTNode::loop_lower)
        .def_ro("target",          &ASTNode::target)
        .def_ro("expr",            &ASTNode::expr)
        .def_ro("accesses",        &ASTNode::accesses)
        .def_ro("target_is_array", &ASTNode::target_is_array)
        .def_ro("condition",       &ASTNode::condition)
        .def_ro("callee",          &ASTNode::callee)
        .def_ro("call_args",       &ASTNode::call_args)
        .def_ro("reduce_src",      &ASTNode::reduce_src)
        .def_ro("reduce_wcr",      &ASTNode::reduce_wcr)
        .def_ro("reduce_identity", &ASTNode::reduce_identity)
        .def_ro("reduce_axes",     &ASTNode::reduce_axes)
        .def_ro("children",        &ASTNode::children)
        .def_ro("else_children",   &ASTNode::else_children)
        .def("__repr__", [](const ASTNode &n) {
            if (n.kind == "loop")
                return std::string("Loop(") + n.loop_iter + "="
                    + std::to_string(n.loop_lower) + ":"
                    + n.loop_bound + ", "
                    + std::to_string(n.children.size()) + " children)";
            if (n.kind == "assign")
                return std::string("Assign(") + n.target + " = "
                    + n.expr.substr(0, 40)
                    + (n.expr.size() > 40 ? "..." : "") + ")";
            if (n.kind == "conditional")
                return std::string("If(") + n.condition + ")";
            if (n.kind == "call")
                return std::string("Call(") + n.callee + ")";
            if (n.kind == "reduce")
                return std::string("Reduce(") + n.target + " = reduce("
                    + n.reduce_src + ", wcr=" + n.reduce_wcr + ")";
            return std::string("<") + n.kind + ">";
        });

    nb::class_<HLFIRModule>(m, "HLFIRModule")
        .def(nb::init<>())
        .def("parse",         &HLFIRModule::parse,
             "Parse HLFIR from a string")
        .def("parse_file",    &HLFIRModule::parse_file,
             "Parse HLFIR from a file path")
        .def("run_passes",    &HLFIRModule::run_passes,
             "Run an mlir-opt-syntax pipeline "
             "(e.g. 'builtin.module(hlfir-propagate-shapes)')")
        .def("dump",          &HLFIRModule::dump,
             "Return the current IR as a string")
        .def("get_variables", &HLFIRModule::get_variables,
             "Classify all hlfir.declare ops -> list[VarInfo]")
        .def("get_ast",       &HLFIRModule::get_ast,
             "Recursive AST of the subroutine body -> list[ASTNode]");
}

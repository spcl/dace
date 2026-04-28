// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Internal cross-TU declarations for the HLFIR AST extraction layer.
// Bodies live in the ast/*.cpp owning each helper.  Public API in
// ast_helpers.h.
#pragma once

#include "bridge/ast/ast_helpers.h"

namespace hlfir_bridge {

ASTNode buildAssignNode(hlfir::AssignOp assign);

ASTNode buildCopyNode(hlfir::AssignOp assign);

ASTNode buildLibCallNode(hlfir::AssignOp assign,
                                mlir::Operation *srcOp,
                                std::string_view callee);

ASTNode buildMemsetNode(hlfir::AssignOp assign);

ASTNode buildReduceNode(hlfir::AssignOp assign, mlir::Operation *redOp,
                               std::string_view wcr,
                               std::string_view identity);

std::vector<ASTNode> buildSectionReduceAssign(
    hlfir::AssignOp assign, hlfir::DesignateOp src,
    std::string_view pyOp, std::string_view identity);

std::vector<ASTNode> buildSectionScalarAssign(
    hlfir::AssignOp assign, hlfir::DesignateOp dst);

std::vector<ASTNode> buildSectionToSectionAssign(
    hlfir::AssignOp assign, hlfir::DesignateOp dst);

ASTNode buildSelectCaseChain(fir::SelectCaseOp sel);

std::vector<ASTNode> buildWholeArrayScalarBroadcast(hlfir::AssignOp assign);

void collectReadAccesses(mlir::Value v,
                         std::vector<AccessInfo> &accesses,
                         int depth);

std::string exprDtypeString(mlir::Type ty);

std::vector<std::string> exprResultShape(mlir::Type ty);

std::string lowerIsPresent(mlir::Value operand);

std::string resolveExtent(mlir::Value shape, unsigned d);

std::string resolveIndex(mlir::Value idx);

std::string scfSynthName(mlir::Value v);

std::vector<ASTNode> walkSCFBeforeRegion(mlir::Block &block);

std::string yieldedExpr(mlir::Value v);

std::vector<ASTNode>
buildMergeLibcall(hlfir::AssignOp assign, hlfir::ElementalOp elem);

std::vector<ASTNode>
buildElementalAssign(hlfir::AssignOp assign, hlfir::ElementalOp elem);

std::vector<ASTNode>
buildElementalCountLibcall(hlfir::AssignOp assign, hlfir::ElementalOp elem);

std::vector<ASTNode>
buildElementalAnyAllReduce(hlfir::AssignOp assign, hlfir::ElementalOp elem,
                           std::string_view wcr,
                           std::string_view identity);

}  // namespace hlfir_bridge

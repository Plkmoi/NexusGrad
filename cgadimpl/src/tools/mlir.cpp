// #include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/AsmState.h"
// #include "mlir/Parser/Parser.h"
// #include "mlir/Support/FileUtilities.h"
// #include "mlir/Support/LogicalResult.h"
// #include "mlir/InitAllDialects.h"
// #include "mlir/InitAllPasses.h"

// // StableHLO modern includes
// #include "stablehlo/dialect/StablehloOps.h"

// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/MemoryBuffer.h"
// #include <iostream>

// using namespace mlir;

// int main() {
//     // Create context and register dialects
//     MLIRContext context;
//     DialectRegistry registry;
//     registerAllDialects(registry);
//     registry.insert<mlir::stablehlo::StablehloDialect>();
//     context.appendDialectRegistry(registry);

//     // Load an .mlir file
//     llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
//         llvm::MemoryBuffer::getFile("example.mlir");
//     if (!fileOrErr) {
//         std::cerr << "Could not open example.mlir\n";
//         return 1;
//     }

//     llvm::SourceMgr sourceMgr;
//     sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

//     // Parse module
//     OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
//     if (!module) {
//         std::cerr << "Failed to parse example.mlir\n";
//         return 1;
//     }

//     // Print module
//     std::cout << "Parsed StableHLO module:\n";
//     module->print(llvm::outs());
//     return 0;
// }
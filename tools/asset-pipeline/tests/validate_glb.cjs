// Usage: node validate_glb.cjs <path-to-glb> <dir-containing-node_modules/gltf-validator>
// Prints the Khronos glTF-Validator report as JSON; exit code 1 if it has errors or warnings.
const fs = require('fs');
const path = require('path');

const [, , glbPath, prefix] = process.argv;
const validator = require(path.join(prefix, 'node_modules', 'gltf-validator'));
const bytes = new Uint8Array(fs.readFileSync(glbPath));

validator.validateBytes(bytes, { maxIssues: 100 }).then((report) => {
  console.log(JSON.stringify(report.issues));
  process.exit(report.issues.numErrors + report.issues.numWarnings > 0 ? 1 : 0);
});

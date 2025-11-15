# Attribution and Licensing Guide for Standalone Server

This guide outlines the proper steps to respect authors and licenses when creating a standalone server based on the `robot_client` velocity controller.

## 📋 BEFORE Copying: License Compliance Checklist

### 1. **Identify All Copyright Holders**

Based on the code analysis, you need to attribute:

- **Franka Robotics GmbH** (2023)
  - Files: `franka_vr_control_client.cpp`, `examples_common.cpp`, `examples_common.h`
  - License: Apache-2.0 (as stated in code comments)

- **Pablo Lopez-Custodio** (2025)
  - Files: `geofik.cpp`, `geofik.h`
  - Source: Adapted from [GeoFIK](https://github.com/PabloLopezCustodio/GeoFIK)
  - Note: Check GeoFIK's license for proper attribution

- **Original Repository Authors**
  - Repository: `franka-vr-teleop` (this repo)
  - Check if there are additional contributors

### 2. **Obtain Full License Texts**

The current `LICENSE` file only contains the disclaimer. You need:

- **Apache-2.0 License Text**: Required for Franka Robotics GmbH code
  - Download from: https://www.apache.org/licenses/LICENSE-2.0
  - Or use standard Apache-2.0 template

- **GeoFIK License**: Check the original GeoFIK repository for its license
  - Repository: https://github.com/PabloLopezCustodio/GeoFIK

### 3. **Verify License Compatibility**

- Apache-2.0 is permissive and allows commercial use
- Ensure GeoFIK's license is compatible with your intended use
- If mixing licenses, clearly separate them in your project

## 📝 DURING Setup: What to Include

### Required Files in Your New Project:

1. **License Files**:
   ```
   LICENSE-APACHE-2.0          # Full Apache-2.0 text for Franka code
   LICENSE-GEOFIK              # GeoFIK license (if different)
   LICENSE                      # Your project's license (if applicable)
   ```

2. **Attribution File** (NOTICES or ATTRIBUTION.md):
   ```
   ATTRIBUTION.md              # Detailed attribution for all components
   ```

3. **Copyright Headers**: Keep all existing copyright headers in source files

4. **README Section**: Add clear attribution section

## ✅ AFTER Copying: Implementation Steps

### Step 1: Create Proper License Files

Create `LICENSE-APACHE-2.0` with full Apache-2.0 text (download from Apache website).

### Step 2: Create Attribution Documentation

Create `ATTRIBUTION.md` with:

```markdown
# Attribution

This project includes code from the following sources:

## Franka Robotics GmbH (2023)
- **Files**: `franka_vr_control_client.cpp`, `examples_common.cpp`, `examples_common.h`
- **License**: Apache-2.0
- **Source**: Adapted from franka-vr-teleop repository
- **Copyright**: Copyright (c) 2023 Franka Robotics GmbH

## GeoFIK - Pablo Lopez-Custodio et al.
- **Files**: `geofik.cpp`, `geofik.h`
- **License**: [Check GeoFIK repository]
- **Source**: https://github.com/PabloLopezCustodio/GeoFIK
- **Author**: Pablo Lopez-Custodio
- **Date**: 2025-02-08
- **Note**: Adapted implementation for Franka robot

## Weighted IK Implementation
- **Files**: `weighted_ik.cpp`, `weighted_ik.h`
- **Note**: Based on GeoFIK with additional optimization
```

### Step 3: Preserve Copyright Headers

**DO NOT REMOVE** existing copyright headers. They should remain in all copied files:

```cpp
// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
```

### Step 4: Update Your Project's README

Add a clear "Attribution" or "Credits" section:

```markdown
## Attribution

This standalone server is built upon excellent work from:

1. **Franka Robotics GmbH** - Original velocity control implementation
   - License: Apache-2.0
   - Source: [Link to original repo if available]

2. **GeoFIK by Pablo Lopez-Custodio et al.** - Inverse kinematics
   - Source: https://github.com/PabloLopezCustodio/GeoFIK
   - Used in: `geofik.cpp`, `weighted_ik.cpp`

See [ATTRIBUTION.md](ATTRIBUTION.md) for detailed information.
```

### Step 5: Add License Information to Build Files

Update your `CMakeLists.txt` or build configuration:

```cmake
# License information
set(PROJECT_LICENSE "Apache-2.0")
set(PROJECT_COPYRIGHT "Copyright (c) 2023 Franka Robotics GmbH; Copyright (c) 2025 Pablo Lopez-Custodio")
```

### Step 6: Include License in Distribution

If you distribute binaries or packages:
- Include `LICENSE-APACHE-2.0` file
- Include `ATTRIBUTION.md`
- Include license information in package metadata

## 🔍 Additional Best Practices

### 1. **Contact Original Authors** (Optional but Recommended)
- Reach out to repository maintainers to:
  - Inform them of your standalone server project
  - Ask if they want to be listed as contributors
  - Verify license understanding

### 2. **Document Your Modifications**
- Clearly mark any changes you make
- Use comments like: `// Modified: [Your name] - [Date] - [Reason]`
- Or maintain a `CHANGELOG.md` for your modifications

### 3. **Link Back to Original**
- In your README, link to the original repository
- Helps others find the source and supports the original authors

### 4. **Consider Contributing Back**
- If you make improvements, consider contributing back via pull request
- This supports the open-source community

## 📄 Example Project Structure

```
your-standalone-server/
├── LICENSE-APACHE-2.0          # Full Apache-2.0 license
├── LICENSE-GEOFIK              # GeoFIK license (if different)
├── ATTRIBUTION.md              # Detailed attribution
├── README.md                   # With attribution section
├── CMakeLists.txt
├── include/
│   ├── examples_common.h       # (with copyright header)
│   ├── geofik.h                # (with copyright header)
│   └── weighted_ik.h
├── src/
│   ├── franka_vr_control_client.cpp  # (with copyright header)
│   ├── examples_common.cpp            # (with copyright header)
│   ├── geofik.cpp                     # (with copyright header)
│   └── weighted_ik.cpp
└── build/
```

## ⚖️ Legal Compliance Summary

**Minimum Requirements:**
- ✅ Keep all copyright notices in source files
- ✅ Include full Apache-2.0 license text
- ✅ Document all attributions clearly
- ✅ Include license files in distribution

**Best Practices:**
- ✅ Contact original authors
- ✅ Link back to original repositories
- ✅ Document your modifications
- ✅ Consider contributing improvements back

## 🚨 Important Notes

1. **The current LICENSE file is incomplete** - It only has the disclaimer. You need the full Apache-2.0 license text.

2. **GeoFIK License** - You must check the original GeoFIK repository to ensure proper compliance with their license terms.

3. **Commercial Use** - Apache-2.0 allows commercial use, but you must still include attribution and license text.

4. **No Warranty** - The "AS IS" disclaimer protects original authors from liability.

## 📚 Resources

- Apache-2.0 License: https://www.apache.org/licenses/LICENSE-2.0
- GeoFIK Repository: https://github.com/PabloLopezCustodio/GeoFIK
- Open Source License Guide: https://opensource.org/licenses

---

**Remember**: Proper attribution is not just legally required—it's a way to honor and support the open-source community that made this work possible! 🙏


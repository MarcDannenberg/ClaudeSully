# sully_engine/kernel_modules/neural_modification.py
# 🧠 Sully's Neural Modification System - Self-improvement capabilities

from typing import Dict, List, Any, Optional, Union, Tuple
import random
import inspect
import importlib
import sys
import os
import json
from datetime import datetime
import re
import difflib
import copy

class CodeRepository:
    """
    Access and manage system code for self-modification.
    Acts as a safe interface to the codebase.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the code repository tracker.
        
        Args:
            base_path: Optional base path to the codebase
        """
        self.base_path = base_path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.module_cache = {}
        self.modification_history = []
        
    def get_module(self, module_name: str) -> str:
        """
        Get the source code of a module.
        
        Args:
            module_name: Name of the module to retrieve
            
        Returns:
            Source code of the module
        """
        # Check if module is in cache
        if module_name in self.module_cache:
            return self.module_cache[module_name]
            
        # Try to find the module file
        module_path = self._find_module_path(module_name)
        if not module_path:
            raise ValueError(f"Module {module_name} not found in codebase")
            
        # Read the module source
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                self.module_cache[module_name] = source_code
                return source_code
        except Exception as e:
            raise IOError(f"Error reading module {module_name}: {str(e)}")
    
    def _find_module_path(self, module_name: str) -> Optional[str]:
        """
        Find the file path for a module.
        
        Args:
            module_name: Name of the module to find
            
        Returns:
            File path or None if not found
        """
        # Check for direct file match
        if module_name.endswith('.py'):
            potential_path = os.path.join(self.base_path, module_name)
            if os.path.exists(potential_path):
                return potential_path
                
        # Check in kernel_modules directory
        kernel_path = os.path.join(self.base_path, 'kernel_modules', f"{module_name}.py")
        if os.path.exists(kernel_path):
            return kernel_path
            
        # Check in main directory
        main_path = os.path.join(self.base_path, f"{module_name}.py")
        if os.path.exists(main_path):
            return main_path
            
        return None
        
    def save_module_variant(self, module_name: str, variant_code: str, variant_name: Optional[str] = None) -> str:
        """
        Save a module variant to the variants directory.
        
        Args:
            module_name: Base module name
            variant_code: The variant code to save
            variant_name: Optional name for the variant
            
        Returns:
            Path to the saved variant
        """
        # Create variants directory if it doesn't exist
        variants_dir = os.path.join(self.base_path, 'variants')
        os.makedirs(variants_dir, exist_ok=True)
        
        # Generate variant name if not provided
        if not variant_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            variant_name = f"{module_name.replace('.py', '')}_{timestamp}"
            
        # Save the variant
        variant_path = os.path.join(variants_dir, f"{variant_name}.py")
        try:
            with open(variant_path, 'w', encoding='utf-8') as f:
                f.write(variant_code)
            return variant_path
        except Exception as e:
            raise IOError(f"Error saving variant: {str(e)}")
            
    def implement_variant(self, module_name: str, variant_code: str, backup: bool = True) -> bool:
        """
        Implement a variant by replacing the existing module code.
        
        Args:
            module_name: Module to replace
            variant_code: New code to implement
            backup: Whether to create a backup
            
        Returns:
            Success indicator
        """
        # Find the module path
        module_path = self._find_module_path(module_name)
        if not module_path:
            raise ValueError(f"Module {module_name} not found")
            
        # Create backup if requested
        if backup:
            backup_path = f"{module_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                with open(module_path, 'r', encoding='utf-8') as src, open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            except Exception as e:
                raise IOError(f"Error creating backup: {str(e)}")
                
        # Implement the new code
        try:
            with open(module_path, 'w', encoding='utf-8') as f:
                f.write(variant_code)
                
            # Update the cache
            self.module_cache[module_name] = variant_code
            
            # Record the modification
            self.modification_history.append({
                "timestamp": datetime.now().isoformat(),
                "module": module_name,
                "module_path": module_path,
                "backup_path": backup_path if backup else None
            })
            
            return True
        except Exception as e:
            raise IOError(f"Error implementing variant: {str(e)}")
            
    def get_modification_history(self) -> List[Dict[str, Any]]:
        """Get the history of code modifications."""
        return self.modification_history


class NeuralModification:
    """
    Advanced system for self-modification and cognitive architecture evolution.
    Enables Sully to analyze, modify, and improve its own code and architecture.
    """

    def __init__(self, reasoning_engine=None, memory_system=None, code_repository=None):
        """
        Initialize the neural modification system.
        
        Args:
            reasoning_engine: Engine for generating modifications
            memory_system: System for tracking performance over time
            code_repository: Repository for accessing and modifying code
        """
        self.reasoning = reasoning_engine
        self.memory = memory_system
        self.code_repository = code_repository or CodeRepository()
        
        self.modification_history = []
        self.current_experiments = {}
        self.performance_metrics = {}
        self.architecture_map = {}
        self.safe_modules = ["neural_modification"]  # Modules that shouldn't modify themselves
        
    def analyze_performance(self, module_name: str, metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze performance of specific modules to identify improvement areas.
        
        Args:
            module_name: Module to analyze
            metrics: Optional performance metrics
            
        Returns:
            Analysis results with improvement suggestions
        """
        # Get module code
        try:
            module_code = self.code_repository.get_module(module_name)
        except Exception as e:
            return {
                "success": False,
                "error": f"Could not access module: {str(e)}",
                "suggestions": []
            }
            
        # Use metrics if provided, otherwise analyze code structure
        if metrics:
            return self._analyze_with_metrics(module_name, module_code, metrics)
        else:
            return self._analyze_code_structure(module_name, module_code)
            
    def _analyze_with_metrics(self, module_name: str, code: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze based on performance metrics."""
        bottlenecks = []
        suggestions = []
        
        # Store metrics for historical comparison
        if module_name not in self.performance_metrics:
            self.performance_metrics[module_name] = []
        self.performance_metrics[module_name].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # Identify performance bottlenecks
        if 'execution_time' in metrics and metrics['execution_time'] > 1.0:
            bottlenecks.append({
                "type": "performance",
                "description": f"Slow execution time: {metrics['execution_time']:.2f}s",
                "severity": "high" if metrics['execution_time'] > 3.0 else "medium"
            })
            
        if 'memory_usage' in metrics and metrics['memory_usage'] > 100 * 1024 * 1024:  # 100 MB
            bottlenecks.append({
                "type": "resource",
                "description": f"High memory usage: {metrics['memory_usage'] / (1024*1024):.2f} MB",
                "severity": "high" if metrics['memory_usage'] > 500 * 1024 * 1024 else "medium"
            })
            
        if 'error_rate' in metrics and metrics['error_rate'] > 0.01:  # 1% error rate
            bottlenecks.append({
                "type": "reliability",
                "description": f"High error rate: {metrics['error_rate'] * 100:.2f}%",
                "severity": "high" if metrics['error_rate'] > 0.05 else "medium"
            })
            
        # Generate suggestions based on bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "performance":
                suggestions.append({
                    "target": "performance",
                    "description": "Consider optimizing algorithm efficiency",
                    "approach": "Identify expensive operations and implement more efficient alternatives"
                })
                
            elif bottleneck["type"] == "resource":
                suggestions.append({
                    "target": "memory",
                    "description": "Reduce memory footprint",
                    "approach": "Implement data streaming or chunking to process data incrementally"
                })
                
            elif bottleneck["type"] == "reliability":
                suggestions.append({
                    "target": "error_handling",
                    "description": "Improve error handling and recovery",
                    "approach": "Add comprehensive exception handling and retry mechanisms"
                })
                
        return {
            "success": True,
            "module": module_name,
            "bottlenecks": bottlenecks,
            "suggestions": suggestions,
            "metrics": metrics
        }
            
    def _analyze_code_structure(self, module_name: str, code: str) -> Dict[str, Any]:
        """Analyze based on code structure and patterns."""
        suggestions = []
        
        # Check for long functions (potential complexity issues)
        long_functions = []
        function_matches = re.finditer(r'def\s+(\w+)\s*\(', code)
        for match in function_matches:
            func_name = match.group(1)
            func_start = match.start()
            
            # Find function end (simplistic approach)
            next_func = code.find('\ndef ', func_start + 1)
            if next_func == -1:
                next_func = len(code)
                
            func_code = code[func_start:next_func]
            func_lines = func_code.count('\n')
            
            # Functions with more than 30 lines might be too complex
            if func_lines > 30:
                long_functions.append({
                    "name": func_name,
                    "lines": func_lines,
                    "start_pos": func_start,
                    "end_pos": next_func
                })
                
        # If long functions found, suggest refactoring
        if long_functions:
            for func in long_functions:
                suggestions.append({
                    "target": "complexity",
                    "description": f"Function '{func['name']}' is {func['lines']} lines long (> 30 recommended)",
                    "approach": "Refactor into smaller, more focused functions"
                })
                
        # Check for error handling
        if 'try:' not in code:
            suggestions.append({
                "target": "reliability",
                "description": "No error handling found in module",
                "approach": "Add try-except blocks for critical operations"
            })
            
        # Check for documentation
        docstring_count = len(re.findall(r'"""[\s\S]*?"""', code))
        function_count = len(re.findall(r'def\s+\w+\s*\(', code))
        class_count = len(re.findall(r'class\s+\w+', code))
        
        # If less docstrings than functions+classes, suggest improving docs
        if docstring_count < (function_count + class_count):
            suggestions.append({
                "target": "documentation",
                "description": f"Documentation coverage: {docstring_count}/{function_count + class_count} functions/classes",
                "approach": "Add or improve docstrings for better code maintainability"
            })
            
        # Check for unused imports
        import_matches = re.finditer(r'import (\w+)|from [\w\.]+ import (\w+)', code)
        imported_modules = []
        for match in import_matches:
            module = match.group(1) or match.group(2)
            if module:
                imported_modules.append(module)
                
        # Check for modules that are imported but not used
        unused_imports = []
        for module in imported_modules:
            # Skip common modules that might be used indirectly
            if module in ['os', 'sys', 'typing', 'datetime']:
                continue
                
            # Check for module usage (simple approach)
            usage_pattern = fr'\b{re.escape(module)}\b'
            usages = re.findall(usage_pattern, code)
            
            # If only one occurrence (the import itself), it might be unused
            if len(usages) <= 1:
                unused_imports.append(module)
                
        if unused_imports:
            suggestions.append({
                "target": "code_quality",
                "description": f"Potentially unused imports: {', '.join(unused_imports)}",
                "approach": "Remove unused imports to improve code clarity"
            })
            
        # Check for overly complex conditionals
        complex_conditionals = re.findall(r'if\s+[^:]+\s+and\s+[^:]+\s+and\s+[^:]+', code)
        if complex_conditionals:
            suggestions.append({
                "target": "complexity",
                "description": f"Found {len(complex_conditionals)} complex conditional statements",
                "approach": "Simplify conditionals by extracting logical parts into separate functions or variables"
            })
            
        # Update architecture map with module structure
        self._update_architecture_map(module_name, code)
            
        return {
            "success": True,
            "module": module_name,
            "suggestions": suggestions,
            "long_functions": long_functions,
            "documentation_coverage": docstring_count / max(1, function_count + class_count),
        }
        
    def _update_architecture_map(self, module_name: str, code: str) -> None:
        """
        Update the architecture map with module structure and dependencies.
        
        Args:
            module_name: Name of the module to analyze
            code: Source code of the module
        """
        # Extract classes and functions
        classes = []
        class_matches = re.finditer(r'class\s+(\w+)(?:\([^)]*\))?:', code)
        for match in class_matches:
            class_name = match.group(1)
            classes.append(class_name)
            
        functions = []
        function_matches = re.finditer(r'def\s+(\w+)\s*\(', code)
        for match in function_matches:
            func_name = match.group(1)
            if not func_name.startswith('_'):  # Only include public functions
                functions.append(func_name)
                
        # Extract imports and dependencies
        imports = []
        import_matches = re.finditer(r'import (\w+)|from ([\w\.]+) import (\w+)', code)
        for match in import_matches:
            if match.group(1):
                # Simple import
                imports.append(match.group(1))
            else:
                # From import
                module = match.group(2)
                item = match.group(3)
                imports.append(f"{module}.{item}")
                
        # Create/update the module entry in the architecture map
        if module_name not in self.architecture_map:
            self.architecture_map[module_name] = {
                "classes": [],
                "functions": [],
                "imports": [],
                "dependencies": [],
                "last_updated": datetime.now().isoformat()
            }
            
        # Update the entry
        self.architecture_map[module_name]["classes"] = classes
        self.architecture_map[module_name]["functions"] = functions
        self.architecture_map[module_name]["imports"] = imports
        self.architecture_map[module_name]["last_updated"] = datetime.now().isoformat()
        
        # Resolve actual module dependencies
        dependencies = []
        for imp in imports:
            # Split the import string
            parts = imp.split('.')
            
            # Resolve internal dependencies (Sully modules)
            potential_modules = [
                parts[0],  # Direct import
                f"{parts[0]}.py",  # With extension
                f"{parts[0]}.{parts[1]}" if len(parts) > 1 else None  # Submodule
            ]
            
            for potential in potential_modules:
                if potential and self.code_repository._find_module_path(potential):
                    dependencies.append(potential)
                    break
                    
        self.architecture_map[module_name]["dependencies"] = dependencies
        
    def get_architecture_map(self, module_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the system architecture map.
        
        Args:
            module_filter: Optional filter for specific module or module pattern
            
        Returns:
            Architecture map dictionary
        """
        if not module_filter:
            return self.architecture_map
        
        # Filter by exact module name
        if module_filter in self.architecture_map:
            return {module_filter: self.architecture_map[module_filter]}
            
        # Filter by pattern
        filtered_map = {}
        pattern = re.compile(module_filter)
        for module_name, module_data in self.architecture_map.items():
            if pattern.search(module_name):
                filtered_map[module_name] = module_data
                
        return filtered_map
        
    def visualize_architecture(self, output_format: str = "text", module_filter: Optional[str] = None) -> str:
        """
        Generate a visualization of the system architecture.
        
        Args:
            output_format: Format for visualization (text, graphviz, json)
            module_filter: Optional filter for specific modules
            
        Returns:
            Visualization in the requested format
        """
        # Get filtered architecture map
        arch_map = self.get_architecture_map(module_filter)
        
        if output_format == "json":
            return json.dumps(arch_map, indent=2)
            
        elif output_format == "graphviz":
            # Generate DOT format for Graphviz
            dot = "digraph Architecture {\n"
            dot += "  rankdir=LR;\n"
            dot += "  node [shape=box, style=filled, fillcolor=lightblue];\n\n"
            
            # Add nodes
            for module_name in arch_map:
                dot += f'  "{module_name}" [label="{module_name}\\n{len(arch_map[module_name]["classes"])} classes, {len(arch_map[module_name]["functions"])} functions"];\n'
                
            # Add edges for dependencies
            for module_name, module_data in arch_map.items():
                for dep in module_data["dependencies"]:
                    if dep in arch_map:
                        dot += f'  "{module_name}" -> "{dep}";\n'
                        
            dot += "}\n"
            return dot
            
        else:  # Default to text format
            text = "SYSTEM ARCHITECTURE\n" + "=" * 20 + "\n\n"
            
            for module_name, module_data in arch_map.items():
                text += f"Module: {module_name}\n"
                text += f"  Classes: {', '.join(module_data['classes']) or 'None'}\n"
                text += f"  Public Functions: {', '.join(module_data['functions']) or 'None'}\n"
                text += f"  Dependencies: {', '.join(module_data['dependencies']) or 'None'}\n"
                text += "\n"
                
            return text
            
    def build_full_architecture_map(self) -> Dict[str, Any]:
        """
        Build a complete architecture map of the entire codebase.
        
        Returns:
            Complete architecture map
        """
        # Reset architecture map
        self.architecture_map = {}
        
        # Get list of all Python files in the codebase
        all_modules = []
        
        # Check main directory
        base_path = self.code_repository.base_path
        for file in os.listdir(base_path):
            if file.endswith('.py'):
                all_modules.append(file)
                
        # Check kernel_modules directory
        kernel_path = os.path.join(base_path, 'kernel_modules')
        if os.path.exists(kernel_path):
            for file in os.listdir(kernel_path):
                if file.endswith('.py'):
                    all_modules.append(file)
                    
        # Analyze each module
        for module_name in all_modules:
            try:
                code = self.code_repository.get_module(module_name)
                self._update_architecture_map(module_name, code)
            except Exception as e:
                # Skip modules that can't be analyzed
                continue
                
        return self.architecture_map
        
    def recommend_improvements(self, module_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Recommend system improvements based on architecture and performance analysis.
        
        Args:
            module_name: Optional specific module to focus on
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        # If no specific module, analyze the overall architecture
        if not module_name:
            # Build full architecture map if not already done
            if not self.architecture_map:
                self.build_full_architecture_map()
                
            # Look for modules with high dependency counts
            dependency_counts = {}
            for mod_name, mod_data in self.architecture_map.items():
                # Count how many other modules depend on this one
                dependents = sum(1 for m, data in self.architecture_map.items() 
                                if mod_name in data["dependencies"])
                dependency_counts[mod_name] = dependents
                
            # Find potential bottleneck modules (high dependents count)
            bottleneck_modules = []
            for mod_name, count in dependency_counts.items():
                if count > 3:  # Arbitrary threshold
                    bottleneck_modules.append((mod_name, count))
                    
            # Sort by dependent count (descending)
            bottleneck_modules.sort(key=lambda x: x[1], reverse=True)
            
            # Add recommendations for high-dependency modules
            for mod_name, count in bottleneck_modules[:3]:  # Top 3
                recommendations.append({
                    "type": "architecture",
                    "target": mod_name,
                    "description": f"High-dependency module with {count} dependents",
                    "suggestion": "Consider refactoring into smaller, more focused modules",
                    "priority": "medium" if count > 5 else "low"
                })
                
            # Look for circular dependencies
            circular_deps = self._detect_circular_dependencies()
            for cycle in circular_deps:
                cycle_str = " -> ".join(cycle)
                recommendations.append({
                    "type": "architecture",
                    "target": cycle[0],  # Pick first module in cycle
                    "description": f"Circular dependency detected: {cycle_str}",
                    "suggestion": "Break circular dependency by introducing an interface or refactoring",
                    "priority": "high"
                })
                
        # Analyze specific module if provided
        if module_name:
            try:
                # Get module code
                code = self.code_repository.get_module(module_name)
                
                # Perform code structure analysis
                analysis = self._analyze_code_structure(module_name, code)
                
                # Convert analysis suggestions to recommendations
                for suggestion in analysis.get("suggestions", []):
                    recommendations.append({
                        "type": "code_quality",
                        "target": module_name,
                        "description": suggestion["description"],
                        "suggestion": suggestion["approach"],
                        "priority": "medium"
                    })
                    
                # Check performance metrics if available
                if module_name in self.performance_metrics:
                    recent_metrics = self.performance_metrics[module_name][-1]
                    
                    # Add recommendations based on metrics
                    if recent_metrics["metrics"].get("execution_time", 0) > 2.0:
                        recommendations.append({
                            "type": "performance",
                            "target": module_name,
                            "description": f"Slow execution: {recent_metrics['metrics']['execution_time']:.2f}s",
                            "suggestion": "Profile code to identify bottlenecks and optimize",
                            "priority": "high"
                        })
                        
                    if recent_metrics["metrics"].get("error_rate", 0) > 0.02:
                        recommendations.append({
                            "type": "reliability",
                            "target": module_name,
                            "description": f"High error rate: {recent_metrics['metrics']['error_rate'] * 100:.2f}%",
                            "suggestion": "Add comprehensive error handling and logging",
                            "priority": "high"
                        })
                        
            except Exception as e:
                # Add recommendation to fix module loading issue
                recommendations.append({
                    "type": "critical",
                    "target": module_name,
                    "description": f"Module analysis failed: {str(e)}",
                    "suggestion": "Fix module loading issue or syntax errors",
                    "priority": "high"
                })
                
        return recommendations
        
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the architecture.
        
        Returns:
            List of circular dependency chains
        """
        def find_cycles(node, visited, path):
            """DFS to find cycles in dependency graph."""
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                return [path[cycle_start:]]
                
            if node in visited:
                return []
                
            visited.add(node)
            path.append(node)
            
            cycles = []
            if node in self.architecture_map:
                for dep in self.architecture_map[node].get("dependencies", []):
                    cycles.extend(find_cycles(dep, visited.copy(), path.copy()))
                    
            return cycles
        
        all_cycles = []
        for module in self.architecture_map:
            cycles = find_cycles(module, set(), [])
            for cycle in cycles:
                if cycle not in all_cycles:
                    all_cycles.append(cycle)
                    
        return all_cycles
        
    def generate_improvement_plan(self, module_name: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a concrete improvement plan for a module.
        
        Args:
            module_name: Module to improve
            issues: List of identified issues
            
        Returns:
            Improvement plan with suggested code changes
        """
        try:
            current_code = self.code_repository.get_module(module_name)
            
            # Group issues by type
            issues_by_type = {}
            for issue in issues:
                issue_type = issue.get("type", "general")
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(issue)
                
            # Generate plan sections
            plan_sections = []
            code_changes = []
            
            # Handle code quality issues
            if "code_quality" in issues_by_type:
                quality_plan = self._generate_code_quality_improvements(
                    module_name, 
                    current_code, 
                    issues_by_type["code_quality"]
                )
                plan_sections.append(quality_plan["description"])
                code_changes.extend(quality_plan["changes"])
                
            # Handle performance issues
            if "performance" in issues_by_type:
                perf_plan = self._generate_performance_improvements(
                    module_name, 
                    current_code, 
                    issues_by_type["performance"]
                )
                plan_sections.append(perf_plan["description"])
                code_changes.extend(perf_plan["changes"])
                
            # Handle reliability issues
            if "reliability" in issues_by_type:
                reliability_plan = self._generate_reliability_improvements(
                    module_name, 
                    current_code, 
                    issues_by_type["reliability"]
                )
                plan_sections.append(reliability_plan["description"])
                code_changes.extend(reliability_plan["changes"])
                
            # Handle architecture issues
            if "architecture" in issues_by_type:
                arch_plan = self._generate_architecture_improvements(
                    module_name, 
                    current_code, 
                    issues_by_type["architecture"]
                )
                plan_sections.append(arch_plan["description"])
                code_changes.extend(arch_plan["changes"])
                
            # Apply changes to create improved code
            improved_code = current_code
            for change in code_changes:
                if "search" in change and "replace" in change:
                    improved_code = improved_code.replace(change["search"], change["replace"])
                    
            # Generate a variant with all changes applied
            variant_path = None
            if code_changes:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                variant_name = f"{module_name.replace('.py', '')}_improved_{timestamp}"
                variant_path = self.code_repository.save_module_variant(
                    module_name, 
                    improved_code, 
                    variant_name
                )
                
            return {
                "module": module_name,
                "issues_addressed": len(issues),
                "plan": "\n\n".join(plan_sections),
                "code_changes": code_changes,
                "improved_code_path": variant_path,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "module": module_name,
                "error": f"Error generating improvement plan: {str(e)}",
                "issues_addressed": 0,
                "plan": f"Could not generate improvement plan: {str(e)}",
                "code_changes": [],
                "timestamp": datetime.now().isoformat()
            }
            
    def _generate_code_quality_improvements(self, module_name: str, code: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate code quality improvements."""
        description = f"## Code Quality Improvements for {module_name}\n\n"
        changes = []
        
        for issue in issues:
            issue_desc = issue.get("description", "Unknown issue")
            suggestion = issue.get("suggestion", "No suggestion provided")
            
            description += f"- **Issue**: {issue_desc}\n"
            description += f"  - **Approach**: {suggestion}\n\n"
            
            # Handle specific issue types
            if "unused imports" in issue_desc.lower():
                # Extract unused imports
                unused_imports = re.findall(r'Potentially unused imports: (.*)', issue_desc)
                if unused_imports:
                    imports = [imp.strip() for imp in unused_imports[0].split(',')]
                    for imp in imports:
                        # Create regex patterns to find import statements
                        patterns = [
                            rf'import {imp}\n',
                            rf'import {imp},',
                            rf'from .* import .*{imp}.*',
                        ]
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, code)
                            for match in matches:
                                # Suggest removal of the import
                                changes.append({
                                    "type": "unused_import",
                                    "search": match,
                                    "replace": "# Removed unused import: " + match if match.endswith('\n') else "# Removed unused import: " + match + '\n',
                                    "description": f"Remove unused import: {imp}"
                                })
                                
            elif "documentation coverage" in issue_desc.lower():
                # Find functions or classes without docstrings
                functions_without_docs = []
                
                # Find classes
                class_matches = re.finditer(r'class\s+(\w+)(?:\([^)]*\))?:', code)
                for match in class_matches:
                    class_name = match.group(1)
                    class_start = match.start()
                    
                    # Check if there's a docstring after the class definition
                    class_code = code[class_start:class_start + 200]  # Look at the first ~200 chars
                    if '"""' not in class_code[:class_code.find('\n\n')]:
                        functions_without_docs.append({
                            "type": "class",
                            "name": class_name,
                            "position": class_start,
                            "needs_docstring": True
                        })
                        
                # Find functions
                func_matches = re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->.*?)?:', code)
                for match in func_matches:
                    func_name = match.group(1)
                    func_params = match.group(2)
                    func_start = match.start()
                    
                    # Check if there's a docstring after the function definition
                    func_end = code.find('\n\n', func_start)
                    if func_end == -1:
                        func_end = len(code)
                    func_code = code[func_start:func_end]
                    
                    if '"""' not in func_code and not func_name.startswith('_'):
                        functions_without_docs.append({
                            "type": "function",
                            "name": func_name,
                            "params": func_params,
                            "position": func_start,
                            "needs_docstring": True
                        })
                        
                # Generate docstring suggestions for each
                for item in functions_without_docs:
                    if item["type"] == "class":
                        indent = re.search(r'^.*class', code[item["position"]-50:item["position"]]).group()[:-5]
                        line_end = code.find(':', item["position"]) + 1
                        
                        # Generate class docstring
                        docstring = f'\n{indent}    """\n{indent}    {item["name"]} class.\n{indent}    """\n'
                        
                        changes.append({
                            "type": "add_docstring",
                            "search": code[item["position"]:line_end],
                            "replace": code[item["position"]:line_end] + docstring,
                            "description": f"Add docstring to class {item['name']}"
                        })
                        
                    elif item["type"] == "function":
                        indent = re.search(r'^.*def', code[item["position"]-50:item["position"]]).group()[:-3]
                        line_end = code.find(':', item["position"]) + 1
                        
                        # Parse parameters
                        params = []
                        for param in item["params"].split(','):
                            param = param.strip()
                            if param and param != 'self':
                                param_name = param.split(':')[0].strip()
                                params.append(param_name)
                                
                        # Generate function docstring
                        docstring = f'\n{indent}    """\n{indent}    {item["name"]} function.\n'
                        
                        # Add param documentation
                        for param in params:
                            docstring += f'{indent}    \n{indent}    Args:\n'
                            for p in params:
                                docstring += f'{indent}        {p}: Description of {p}\n'
                            break  # Only add once
                            
                        docstring += f'{indent}    """\n'
                        
                        changes.append({
                            "type": "add_docstring",
                            "search": code[item["position"]:line_end],
                            "replace": code[item["position"]:line_end] + docstring,
                            "description": f"Add docstring to function {item['name']}"
                        })
                        
            elif "long function" in issue_desc.lower():
                # Find the function name
                match = re.search(r"Function '(\w+)' is (\d+) lines", issue_desc)
                if match:
                    func_name = match.group(1)
                    
                    # Find the function in code
                    func_match = re.search(rf'def\s+{func_name}\s*\(([^)]*)\)(?:\s*->.*?)?:', code)
                    if func_match:
                        # Simply add a comment to highlight this for refactoring
                        # (Actual refactoring would require deeper analysis)
                        changes.append({
                            "type": "highlight_long_function",
                            "search": func_match.group(0),
                            "replace": f"# TODO: Refactor this long function into smaller units\n{func_match.group(0)}",
                            "description": f"Highlight function {func_name} for refactoring"
                        })
                        
                        description += f"  - **Note**: Automatic refactoring of long functions requires deeper analysis. Added TODO comment for manual review.\n\n"
                        
        return {
            "description": description,
            "changes": changes
        }
                
    def _generate_performance_improvements(self, module_name: str, code: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance improvements."""
        description = f"## Performance Improvements for {module_name}\n\n"
        changes = []
        
        for issue in issues:
            issue_desc = issue.get("description", "Unknown issue")
            suggestion = issue.get("suggestion", "No suggestion provided")
            
            description += f"- **Issue**: {issue_desc}\n"
            description += f"  - **Approach**: {suggestion}\n\n"
            
            # Add performance profiling imports if not present
            if "profile code" in suggestion.lower() and "import cProfile" not in code:
                changes.append({
                    "type": "add_profiling",
                    "search": "import ",  # Find the first import
                    "replace": "import cProfile\nimport pstats\nimport io\nimport ",
                    "description": "Add profiling imports"
                })
                
                # Add profiling utility function
                profile_func = """
def profile_function(func):
    \"\"\"
    Decorator to profile a function.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    \"\"\"
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Print profiling results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Print top 10 time-consuming functions
        print(s.getvalue())
        
        return result
    return wrapper
"""
                # Find a good place to add the function
                match = re.search(r'class \w+:', code)
                if match:
                    # Add before the first class
                    changes.append({
                        "type": "add_profiling_decorator",
                        "search": match.group(0),
                        "replace": profile_func + "\n\n" + match.group(0),
                        "description": "Add profiling decorator function"
                    })
                    
                    description += "  - **Added**: Profiling decorator to easily profile individual functions\n"
                    description += "  - **Usage**: Add `@profile_function` before any function you want to profile\n\n"
                
        return {
            "description": description,
            "changes": changes
        }
        
    def _generate_reliability_improvements(self, module_name: str, code: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate reliability improvements."""
        description = f"## Reliability Improvements for {module_name}\n\n"
        changes = []
        
        for issue in issues:
            issue_desc = issue.get("description", "Unknown issue")
            suggestion = issue.get("suggestion", "No suggestion provided")
            
            description += f"- **Issue**: {issue_desc}\n"
            description += f"  - **Approach**: {suggestion}\n\n"
            
            # Add logging if not present
            if "error handling" in suggestion.lower() and "import logging" not in code:
                changes.append({
                    "type": "add_logging",
                    "search": "import ",  # Find the first import
                    "replace": "import logging\nimport ",
                    "description": "Add logging import"
                })
                
                # Add logger initialization
                logger_init = """
# Configure logging
logger = logging.getLogger(__name__)
"""
                # Find a good place to add the logger
                imports_end = 0
                for match in re.finditer(r'^import|^from', code, re.MULTILINE):
                    potential_end = code.find('\n\n', match.start())
                    if potential_end > imports_end:
                        imports_end = potential_end
                
                if imports_end > 0:
                    changes.append({
                        "type": "add_logger",
                        "search": code[imports_end:imports_end+2],
                        "replace": logger_init + code[imports_end:imports_end+2],
                        "description": "Add logger initialization"
                    })
                    
                    description += "  - **Added**: Logging configuration\n"
                
            # Find functions without try-except
            if "error handling" in suggestion.lower():
                # Find public functions
                func_matches = re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->.*?)?:', code)
                for match in func_matches:
                    func_name = match.group(1)
                    
                    # Skip magic and private methods
                    if func_name.startswith('_'):
                        continue
                        
                    # Find function body
                    func_start = match.end()
                    next_func = code.find('\ndef ', func_start)
                    if next_func == -1:
                        next_func = len(code)
                    func_body = code[func_start:next_func]
                    
                    # Skip if already has try-except
                    if 'try:' in func_body:
                        continue
                        
                    # Find function indentation
                    indentation = ""
                    lines = func_body.split('\n')
                    for line in lines:
                        if line.strip():
                            indentation = re.match(r'^\s*', line).group()
                            break
                            
                    # Create try-except wrapper
                    original_body = func_body
                    try_except_body = "\n" + indentation + "try:\n"
                    
                    # Indent original body
                    for line in original_body.split('\n'):
                        if line.strip():
                            try_except_body += indentation + "    " + line.lstrip() + "\n"
                        else:
                            try_except_body += "\n"
                            
                    # Add except block
                    try_except_body += indentation + "except Exception as e:\n"
                    try_except_body += indentation + "    logger.error(f\"Error in {func_name}: {str(e)}\")\n"
                    try_except_body += indentation + "    raise\n"
                    
                    # Create change
                    changes.append({
                        "type": "add_error_handling",
                        "search": original_body,
                        "replace": try_except_body,
                        "description": f"Add error handling to function {func_name}"
                    })
                    
        return {
            "description": description,
            "changes": changes
        }
        
    def _generate_architecture_improvements(self, module_name: str, code: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate architecture improvements."""
        description = f"## Architecture Improvements for {module_name}\n\n"
        changes = []
        
        for issue in issues:
            issue_desc = issue.get("description", "Unknown issue")
            suggestion = issue.get("suggestion", "No suggestion provided")
            
            description += f"- **Issue**: {issue_desc}\n"
            description += f"  - **Approach**: {suggestion}\n\n"
            
            # Handle circular dependencies
            if "circular dependency" in issue_desc.lower():
                # Extract the cycle
                cycle_match = re.search(r'Circular dependency detected: (.*)', issue_desc)
                if cycle_match:
                    cycle = cycle_match.group(1)
                    description += f"  - **Note**: Detected circular dependency in: {cycle}\n"
                    description += "  - **Resolution Strategy**: Extract common functionality into a shared module\n\n"
                    
                    # Add comment at the top of the file
                    changes.append({
                        "type": "highlight_circular_dependency",
                        "search": "import ",  # Find the first import
                        "replace": f"# WARNING: This module is part of a circular dependency: {cycle}\n# TODO: Refactor to extract shared functionality into a separate module\n\nimport ",
                        "description": "Highlight circular dependency for refactoring"
                    })
                    
            # Handle high-dependency modules
            if "high-dependency module" in issue_desc.lower():
                # Add comment at the top of the file
                changes.append({
                    "type": "highlight_high_dependency",
                    "search": "import ",  # Find the first import
                    "replace": f"# NOTE: This is a high-dependency module with many dependent modules\n# TODO: Consider refactoring into smaller, more focused modules\n\nimport ",
                    "description": "Highlight high-dependency module for refactoring"
                })
                
                description += "  - **Note**: This is a more complex architectural issue that requires manual intervention.\n"
                description += "  - **Resolution Strategy**: Review module responsibilities and consider:\n"
                description += "    - Extracting independent functionality into separate modules\n"
                description += "    - Creating interface modules to decouple dependencies\n"
                description += "    - Implementing a dependency injection pattern\n\n"
                
        return {
            "description": description,
            "changes": changes
        }
        
    def analyze_module(self, module_name: str) -> Dict[str, Any]:
        """
        Comprehensively analyze a specific module.
        
        Args:
            module_name: Module to analyze
            
        Returns:
            Analysis results
        """
        try:
            # Get module code
            code = self.code_repository.get_module(module_name)
            
            # Perform code structure analysis
            structure_analysis = self._analyze_code_structure(module_name, code)
            
            # Check performance metrics if available
            performance_data = None
            if module_name in self.performance_metrics and self.performance_metrics[module_name]:
                performance_data = self.performance_metrics[module_name][-1]
                
            # Get dependencies information
            dependencies = []
            dependents = []
            
            # Update architecture map if needed
            if module_name not in self.architecture_map:
                self._update_architecture_map(module_name, code)
                
            # Get direct dependencies
            if module_name in self.architecture_map:
                dependencies = self.architecture_map[module_name].get("dependencies", [])
                
            # Find modules that depend on this one
            for mod_name, mod_data in self.architecture_map.items():
                if module_name in mod_data.get("dependencies", []):
                    dependents.append(mod_name)
                    
            # Compile the full analysis
            return {
                "module": module_name,
                "timestamp": datetime.now().isoformat(),
                "code_analysis": structure_analysis,
                "performance_metrics": performance_data,
                "dependencies": {
                    "depends_on": dependencies,
                    "dependents": dependents
                },
                "recommendations": self.recommend_improvements(module_name)
            }
            
        except Exception as e:
            return {
                "module": module_name,
                "timestamp": datetime.now().isoformat(),
                "error": f"Analysis failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
            
    def generate_experimental_variant(self, module_name: str, experiment_type: str) -> Dict[str, Any]:
        """
        Generate an experimental variant of a module.
        
        Args:
            module_name: Module to experiment with
            experiment_type: Type of experiment to perform
            
        Returns:
            Experiment details with variant path
        """
        # Make sure the module is not in the safe list
        if any(safe_mod in module_name for safe_mod in self.safe_modules):
            return {
                "success": False,
                "error": f"Cannot experiment with {module_name} as it's in the safe list"
            }
            
        try:
            # Get module code
            code = self.code_repository.get_module(module_name)
            
            # Different experiment types
            if experiment_type == "performance":
                variant_code, changes = self._performance_experiment(module_name, code)
                description = "Performance optimization experiment"
            elif experiment_type == "reliability":
                variant_code, changes = self._reliability_experiment(module_name, code)
                description = "Reliability enhancement experiment"
            elif experiment_type == "code_quality":
                variant_code, changes = self._code_quality_experiment(module_name, code)
                description = "Code quality improvement experiment"
            else:
                return {
                    "success": False,
                    "error": f"Unknown experiment type: {experiment_type}"
                }
                
            # Generate unique experiment ID
            experiment_id = f"{module_name}_{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save the variant
            variant_path = self.code_repository.save_module_variant(
                module_name, 
                variant_code, 
                f"{experiment_id}_variant"
            )
            
            # Record the experiment
            self.current_experiments[experiment_id] = {
                "module": module_name,
                "type": experiment_type,
                "timestamp": datetime.now().isoformat(),
                "variant_path": variant_path,
                "changes": changes,
                "status": "created",
                "results": None
            }
            
            return {
                "success": True,
                "experiment_id": experiment_id,
                "module": module_name,
                "type": experiment_type,
                "description": description,
                "variant_path": variant_path,
                "changes": changes
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Experiment generation failed: {str(e)}",
                "module": module_name,
                "type": experiment_type
            }
            
    def _performance_experiment(self, module_name: str, code: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a performance-optimized variant.
        
        Args:
            module_name: Module name
            code: Module source code
            
        Returns:
            Tuple of (modified code, list of changes)
        """
        changes = []
        modified_code = code
        
        # Add profiling imports if not present
        if "import cProfile" not in code:
            modified_code = "import cProfile\nimport pstats\nimport io\n" + modified_code
            changes.append({
                "type": "add_imports",
                "description": "Add profiling imports"
            })
            
        # Find expensive loops and add optimization
        loop_matches = re.finditer(r'for\s+(\w+)\s+in\s+([^:]+):', modified_code)
        loops_optimized = 0
        
        for match in loop_matches:
            iterator = match.group(1)
            iterable = match.group(2)
            
            # Skip already optimized loops
            if "list(" in iterable:
                continue
                
            # Only optimize certain iterables
            if any(func in iterable for func in ["range", "filter", "map", "zip"]):
                # Optimize by converting to list
                new_line = f"for {iterator} in list({iterable}):"
                modified_code = modified_code.replace(match.group(0), new_line)
                
                changes.append({
                    "type": "optimize_loop",
                    "iterator": iterator,
                    "iterable": iterable,
                    "description": f"Pre-compute iterable '{iterable}' to avoid repeated evaluation"
                })
                
                loops_optimized += 1
                if loops_optimized >= 3:  # Limit optimizations
                    break
                    
        # Look for repeated dict lookups in loops
        dict_lookup_pattern = r'for\s+\w+\s+in\s+[^:]+:\s*[^\n]*?(\w+)\[([^\]]+)\]'
        dict_lookups = re.finditer(dict_lookup_pattern, modified_code, re.DOTALL)
        
        for match in dict_lookups:
            dict_name = match.group(1)
            key_expr = match.group(2)
            
            # Only optimize if the key isn't dynamic
            if not re.search(r'\w+\[', key_expr):
                full_match = match.group(0)
                loop_start = full_match.find("for")
                loop_header_end = full_match.find(":", loop_start)
                
                # Get indentation
                indentation = ""
                for c in reversed(full_match[:loop_start]):
                    if c == '\n':
                        break
                    indentation = c + indentation
                    
                # Create caching snippet
                cache_var = f"{dict_name}_cached"
                cache_snippet = f"{indentation}{cache_var} = {dict_name}\n"
                
                # Replace dict reference with cache
                modified_loop = full_match.replace(f"{dict_name}[", f"{cache_var}[")
                
                # Insert cache before loop
                modified_code = modified_code.replace(full_match, cache_snippet + modified_loop)
                
                changes.append({
                    "type": "optimize_dict_lookup",
                    "dict": dict_name,
                    "description": f"Cache dictionary '{dict_name}' reference to reduce lookup overhead"
                })
                
        # Add profiling decorator
        profile_func = """
def profile_function(func):
    \"\"\"
    Decorator to profile a function.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    \"\"\"
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Print profiling results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Print top 10 time-consuming functions
        print(s.getvalue())
        
        return result
    return wrapper

"""
        # Add decorator after imports
        import_end = 0
        for pattern in ["import ", "from "]:
            last_import = modified_code.rfind(pattern)
            if last_import > import_end:
                # Find end of import block
                block_end = modified_code.find("\n\n", last_import)
                if block_end > import_end:
                    import_end = block_end
                    
        if import_end > 0:
            modified_code = modified_code[:import_end+2] + profile_func + modified_code[import_end+2:]
            changes.append({
                "type": "add_profiling_decorator",
                "description": "Add profiling decorator for performance testing"
            })
            
            # Add decorator to likely performance-critical functions
            # Look for functions with loops or complex operations
            func_pattern = r'def\s+(\w+)\s*\([^)]*\):'
            for match in re.finditer(func_pattern, modified_code):
                func_name = match.group(1)
                
                # Skip private/internal functions
                if func_name.startswith('_'):
                    continue
                    
                # Find function body
                func_start = match.start()
                func_def = match.group(0)
                
                # Check if function contains loops or other expensive operations
                next_func = modified_code.find('\ndef ', func_start + 1)
                if next_func == -1:
                    next_func = len(modified_code)
                    
                func_body = modified_code[func_start:next_func]
                
                if any(pattern in func_body for pattern in ['for ', 'while ', '.join(', '.sort(', 'sorted(']):
                    # Add profiling decorator
                    indentation = ""
                    for c in reversed(modified_code[:func_start]):
                        if c == '\n':
                            break
                        indentation = c + indentation
                        
                    modified_code = modified_code.replace(
                        func_def,
                        f"{indentation}@profile_function\n{func_def}"
                    )
                    
                    changes.append({
                        "type": "profile_function",
                        "function": func_name,
                        "description": f"Add profiling to potentially expensive function '{func_name}'"
                    })
                    
        return modified_code, changes
        
    def _reliability_experiment(self, module_name: str, code: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a reliability-enhanced variant.
        
        Args:
            module_name: Module name
            code: Module source code
            
        Returns:
            Tuple of (modified code, list of changes)
        """
        changes = []
        modified_code = code
        
        # Add logging imports if not present
        if "import logging" not in code:
            modified_code = "import logging\n" + modified_code
            changes.append({
                "type": "add_imports",
                "description": "Add logging imports"
            })
            
            # Add logger initialization
            logger_init = """
# Configure logging
logger = logging.getLogger(__name__)
"""
            # Find a good place to add the logger
            imports_end = 0
            for match in re.finditer(r'^import|^from', modified_code, re.MULTILINE):
                potential_end = modified_code.find('\n\n', match.start())
                if potential_end > imports_end:
                    imports_end = potential_end
            
            if imports_end > 0:
                modified_code = modified_code[:imports_end+2] + logger_init + modified_code[imports_end+2:]
                changes.append({
                    "type": "add_logger",
                    "description": "Add logger initialization"
                })
                
        # Find public methods without try-except
        methods_wrapped = 0
        func_matches = re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->.*?)?:', modified_code)
        for match in re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->.*?)?:', code):
            func_name = match.group(1)
            
            # Skip magic and private methods
            if func_name.startswith('_'):
                continue
                
            # Find function body
            func_start = match.end()
            next_func = code.find('\ndef ', func_start)
            if next_func == -1:
                next_func = len(code)
            func_body = code[func_start:next_func]
            
            # Skip if already has try-except
            if 'try:' in func_body:
                continue
                
            # Extract indentation
            indentation = ""
            first_line = re.search(r'\n\s+', func_body)
            if first_line:
                indentation = first_line.group(0)[1:]  # Skip the newline
                
            # Split into lines, preserving empty lines
            lines = []
            for line in func_body.split('\n'):
                if line.strip():
                    # Non-empty line: indent it further for try block
                    if line.startswith(indentation):
                        lines.append(indentation + "    " + line[len(indentation):])
                    else:
                        lines.append(indentation + "    " + line)
                else:
                    lines.append("")
                    
            # Build the new body with try-except
            new_body = f"\n{indentation}try:\n" + "\n".join(lines)
            new_body += f"\n{indentation}except Exception as e:\n"
            new_body += f"{indentation}    logger.error(f\"Error in {func_name}: {{str(e)}}\")\n"
            new_body += f"{indentation}    raise\n"
            
            # Replace the old function body with the new try-except version
            modified_code = modified_code.replace(func_body, new_body)
            
            changes.append({
                "type": "add_error_handling",
                "function": func_name,
                "description": f"Add try-except block to function '{func_name}'"
            })
            
            methods_wrapped += 1
            if methods_wrapped >= 5:  # Limit the number of methods to wrap
                break
                
        # Add input validation to function parameters
        methods_validated = 0
        for match in re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->.*?)?:', modified_code):
            func_name = match.group(1)
            params = match.group(2)
            
            # Skip magic, private methods, and methods with no params
            if func_name.startswith('_') or not params.strip() or 'self' == params.strip():
                continue
                
            # Skip if already has validation
            func_start = match.end()
            next_func = modified_code.find('\ndef ', func_start)
            if next_func == -1:
                next_func = len(modified_code)
            func_body = modified_code[func_start:next_func]
            
            if 'if ' in func_body[:100] and any(p.strip().split(':')[0].strip() in func_body[:100] for p in params.split(',')):
                continue  # Already has some validation
                
            # Extract indentation
            indentation = ""
            first_line = re.search(r'\n\s+', func_body)
            if first_line:
                indentation = first_line.group(0)[1:]  # Skip the newline
                
            # Generate validation for each parameter
            validation_lines = []
            for param in params.split(','):
                param = param.strip()
                if param == 'self' or not param or '=' in param:
                    continue  # Skip self, empty params, and params with defaults
                
                param_name = param.split(':')[0].strip()
                
                # Generate validation based on parameter name hints
                if 'path' in param_name or 'file' in param_name:
                    validation_lines.append(f"{indentation}if not os.path.exists({param_name}):")
                    validation_lines.append(f"{indentation}    logger.error(f\"File not found: {{{param_name}}}\")")
                    validation_lines.append(f"{indentation}    raise FileNotFoundError(f\"File not found: {{{param_name}}}\")")
                elif 'list' in param_name or param_name.endswith('s'):
                    validation_lines.append(f"{indentation}if not isinstance({param_name}, (list, tuple)):")
                    validation_lines.append(f"{indentation}    logger.error(f\"Expected list/tuple for {param_name}, got {{type({param_name})}}\")")
                    validation_lines.append(f"{indentation}    raise TypeError(f\"Expected list/tuple for {param_name}, got {{type({param_name})}}\")")
                elif 'dict' in param_name or 'map' in param_name:
                    validation_lines.append(f"{indentation}if not isinstance({param_name}, dict):")
                    validation_lines.append(f"{indentation}    logger.error(f\"Expected dict for {param_name}, got {{type({param_name})}}\")")
                    validation_lines.append(f"{indentation}    raise TypeError(f\"Expected dict for {param_name}, got {{type({param_name})}}\")")
                else:
                    validation_lines.append(f"{indentation}if {param_name} is None:")
                    validation_lines.append(f"{indentation}    logger.error(\"{param_name} cannot be None\")")
                    validation_lines.append(f"{indentation}    raise ValueError(\"{param_name} cannot be None\")")
                    
            if validation_lines:
                # Add the validation code at the start of the function
                validation_code = '\n'.join(validation_lines) + '\n'
                
                # Find where to insert the validation (after docstring if present)
                docstring_end = func_body.find('"""', 3)
                if '"""' in func_body[:5] and docstring_end > 5:
                    docstring_end = func_body.find('\n', docstring_end)
                    new_body = func_body[:docstring_end+1] + '\n' + validation_code + func_body[docstring_end+1:]
                else:
                    new_body = '\n' + validation_code + func_body[1:]  # Skip first newline
                    
                # Replace the old function body with the new validated version
                modified_code = modified_code.replace(func_body, new_body)
                
                changes.append({
                    "type": "add_validation",
                    "function": func_name,
                    "description": f"Add parameter validation to function '{func_name}'"
                })
                
                methods_validated += 1
                if methods_validated >= 3:  # Limit validations
                    break
                    
        # Add retry mechanism for external operations
        retry_decorator = """
def with_retry(max_attempts=3, retry_delay=1.0):
    \"\"\"
    Decorator to retry a function in case of exceptions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Wrapped function with retry logic
    \"\"\"
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt+1}/{max_attempts} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        time.sleep(retry_delay)
            
            # If we get here, all attempts failed
            logger.error(f"All {max_attempts} attempts failed")
            raise last_exception
        return wrapper
    return decorator
"""
        
        # Add retry decorator after imports and before first class/function
        first_def = modified_code.find('\ndef ')
        first_class = modified_code.find('\nclass ')
        
        insertion_point = min(pos for pos in [first_def, first_class] if pos > 0)
        if insertion_point > 0:
            modified_code = modified_code[:insertion_point] + '\n' + retry_decorator + modified_code[insertion_point:]
            
            changes.append({
                "type": "add_retry_decorator",
                "description": "Add retry decorator for improved robustness"
            })
            
            # Apply retry decorator to likely external/IO operations
            for match in re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->.*?)?:', modified_code):
                func_name = match.group(1)
                
                # Skip decorated or private methods
                if func_name.startswith('_') or '@' in modified_code[max(0, match.start()-50):match.start()]:
                    continue
                    
                # Look for functions that might involve external operations
                if any(pattern in func_name.lower() for pattern in ['load', 'save', 'read', 'write', 'fetch', 'download', 'upload', 'connect']):
                    # Add retry decorator
                    func_pos = match.start()
                    indentation = ""
                    
                    # Get indentation
                    for c in reversed(modified_code[:func_pos]):
                        if c == '\n':
                            break
                        indentation = c + indentation
                        
                    # Add decorator
                    decorator_line = f"{indentation}@with_retry(max_attempts=3, retry_delay=1.0)\n"
                    modified_code = modified_code[:func_pos] + decorator_line + modified_code[func_pos:]
                    
                    changes.append({
                        "type": "add_retry",
                        "function": func_name,
                        "description": f"Add retry logic to potentially external-facing function '{func_name}'"
                    })
                    
                    # Only add to a couple of functions
                    if len([c for c in changes if c["type"] == "add_retry"]) >= 2:
                        break
            
        return modified_code, changes
        
    def _code_quality_experiment(self, module_name: str, code: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a code quality improved variant.
        
        Args:
            module_name: Module name
            code: Module source code
            
        Returns:
            Tuple of (modified code, list of changes)
        """
        changes = []
        modified_code = code
        
        # Add type hints
        functions_with_type_hints = 0
        func_matches = re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->.*?)?:', modified_code)
        for match in func_matches:
            func_name = match.group(1)
            params = match.group(2)
            return_type = match.group(0).split(':')[0]
            
            # Skip already typed functions
            if '->' in return_type or func_name.startswith('_'):
                continue
                
            # Extract parameter list
            param_list = [p.strip() for p in params.split(',') if p.strip()]
            
            # Skip if already has type hints
            if any(':' in p for p in param_list):
                continue
                
            # Generate type hints based on parameter names
            typed_params = []
            for param in param_list:
                if param == 'self':
                    typed_params.append('self')
                    continue
                    
                # Infer type from parameter name
                param_name = param.split('=')[0].strip()
                
                if 'path' in param_name or 'file' in param_name or 'name' in param_name:
                    typed_params.append(f"{param_name}: str")
                elif 'list' in param_name or param_name.endswith('s'):
                    typed_params.append(f"{param_name}: List[Any]")
                elif 'dict' in param_name or 'map' in param_name:
                    typed_params.append(f"{param_name}: Dict[str, Any]")
                elif 'flag' in param_name or 'enable' in param_name:
                    typed_params.append(f"{param_name}: bool")
                elif 'count' in param_name or 'num' in param_name or 'index' in param_name:
                    typed_params.append(f"{param_name}: int")
                elif 'rate' in param_name or 'value' in param_name:
                    typed_params.append(f"{param_name}: float")
                else:
                    typed_params.append(f"{param_name}: Any")
                    
            # Infer return type from function name and body
            return_hint = "None"  # Default
            if "get_" in func_name or "fetch_" in func_name or "find_" in func_name:
                return_hint = "Any"
            elif "is_" in func_name or "has_" in func_name or "check_" in func_name:
                return_hint = "bool"
            elif "count_" in func_name:
                return_hint = "int"
            
            # Create the new function signature
            old_sig = match.group(0)
            new_sig = f"def {func_name}({', '.join(typed_params)}) -> {return_hint}:"
            
            # Replace the function signature
            modified_code = modified_code.replace(old_sig, new_sig)
            
            changes.append({
                "type": "add_type_hints",
                "function": func_name,
                "description": f"Add type hints to function '{func_name}'"
            })
            
            functions_with_type_hints += 1
            if functions_with_type_hints >= 3:  # Limit the number of functions to modify
                break
                
        # Add doc comments
        functions_with_docs = 0
        for match in re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->.*?)?:', modified_code):
            func_name = match.group(1)
            params = match.group(2)
            
            # Skip private methods
            if func_name.startswith('_'):
                continue
                
            # Find function body
            func_start = match.end()
            next_func = modified_code.find('\ndef ', func_start)
            if next_func == -1:
                next_func = len(modified_code)
            func_body = modified_code[func_start:next_func]
            
            # Skip if already has docstring
            if '"""' in func_body[:50]:
                continue
                
            # Extract parameter list
            param_list = [p.strip() for p in params.split(',') if p.strip() and p.strip() != 'self']
            
            # Extract return type if present
            return_type = "None"
            sig_match = re.search(r'def\s+\w+\s*\([^)]*\)\s*->\s*(\w+)', modified_code[match.start():match.end()])
            if sig_match:
                return_type = sig_match.group(1)
                
            # Generate docstring
            indentation = ""
            first_line = re.search(r'\n\s+', func_body)
            if first_line:
                indentation = first_line.group(0)[1:]  # Skip the newline
                
            docstring = f"\n{indentation}\"\"\"\n"
            docstring += f"{indentation}{func_name.replace('_', ' ').capitalize()}.\n\n"
            
            # Add parameters
            if param_list:
                docstring += f"{indentation}Args:\n"
                for param in param_list:
                    param_name = param.split(':')[0].strip().split('=')[0].strip()
                    docstring += f"{indentation}    {param_name}: Description of {param_name}\n"
                docstring += "\n"
                
            # Add return value
            if return_type != "None":
                docstring += f"{indentation}Returns:\n"
                docstring += f"{indentation}    {return_type}\n"
                
            docstring += f"{indentation}\"\"\"\n"
            
            # Insert docstring at the beginning of the function
            new_body = func_body[:1] + docstring + func_body[1:]
            modified_code = modified_code.replace(func_body, new_body)
            
            changes.append({
                "type": "add_docstring",
                "function": func_name,
                "description": f"Add docstring to function '{func_name}'"
            })
            
            functions_with_docs += 1
            if functions_with_docs >= 3:  # Limit the number of functions to modify
                break
                
        # Improve code formatting
        # Fix line spacing
        modified_code = re.sub(r'\n\n\n+', '\n\n', modified_code)  # Replace 3+ newlines with 2
        changes.append({
            "type": "improve_formatting",
            "description": "Normalize line spacing"
        })
        
        # Add type imports if needed
        if 'List[' in modified_code or 'Dict[' in modified_code or 'Any' in modified_code:
            if 'from typing import ' not in modified_code:
                modified_code = "from typing import Dict, List, Any, Optional, Union, Tuple\n" + modified_code
                changes.append({
                    "type": "add_imports",
                    "description": "Add typing imports"
                })
            elif 'from typing import ' in modified_code and not all(t in modified_code for t in ['Dict', 'List', 'Any']):
                # Find existing typing import
                typing_import = re.search(r'from typing import (.*)', modified_code).group(0)
                # Add missing types
                new_types = []
                for t in ['Dict', 'List', 'Any', 'Optional']:
                    if t not in typing_import:
                        new_types.append(t)
                
                if new_types:
                    new_typing_import = typing_import.rstrip() + ', ' + ', '.join(new_types)
                    modified_code = modified_code.replace(typing_import, new_typing_import)
                    changes.append({
                        "type": "update_imports",
                        "description": f"Add missing typing imports: {', '.join(new_types)}"
                    })
                    
        # Remove unused imports
        import_lines = re.findall(r'^(?:import|from)\s+(\w+)(?:\s+import\s+.+)?
            , modified_code, re.MULTILINE)
        for imp in import_lines:
            # Skip common modules
            if imp in ['os', 'sys', 'typing', 'datetime', 're', 'json']:
                continue
                
            # Check if import is used
            usage_count = len(re.findall(rf'\b{imp}\b', modified_code))
            if usage_count <= 1:  # Only appears in the import statement
                # Find the import line
                import_line = re.search(rf'^(?:import|from)\s+{imp}(?:\s+import\s+.+)?
            , modified_code, re.MULTILINE)
                if import_line:
                    # Comment out the import
                    old_line = import_line.group(0)
                    new_line = f"# Unused import: {old_line}"
                    modified_code = modified_code.replace(old_line, new_line)
                    
                    changes.append({
                        "type": "remove_unused_import",
                        "import": imp,
                        "description": f"Comment out unused import: {imp}"
                    })
        
        return modified_code, changes
        
    def record_experiment_results(self, experiment_id: str, results: Dict[str, Any], success: bool = True) -> Dict[str, Any]:
        """
        Record results from an experiment.
        
        Args:
            experiment_id: ID of the experiment
            results: Results data
            success: Whether the experiment was successful
            
        Returns:
            Updated experiment data
        """
        if experiment_id not in self.current_experiments:
            return {
                "success": False,
                "error": f"Experiment {experiment_id} not found"
            }
            
        # Update experiment data
        self.current_experiments[experiment_id].update({
            "status": "completed" if success else "failed",
            "results": results,
            "completion_time": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "experiment": self.current_experiments[experiment_id]
        }
        
    def implement_experiment(self, experiment_id: str, backup: bool = True) -> Dict[str, Any]:
        """
        Implement an experimental variant as the main module.
        
        Args:
            experiment_id: ID of the experiment to implement
            backup: Whether to create a backup
            
        Returns:
            Implementation result
        """
        if experiment_id not in self.current_experiments:
            return {
                "success": False,
                "error": f"Experiment {experiment_id} not found"
            }
            
        experiment = self.current_experiments[experiment_id]
        
        # Check if experiment was successful
        if experiment["status"] != "completed":
            return {
                "success": False,
                "error": f"Experiment {experiment_id} not completed successfully"
            }
            
        try:
            module_name = experiment["module"]
            
            # Load the variant code
            with open(experiment["variant_path"], 'r', encoding='utf-8') as f:
                variant_code = f.read()
                
            # Implement the variant
            success = self.code_repository.implement_variant(module_name, variant_code, backup)
            
            if success:
                # Record implementation in history
                self.modification_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "module": module_name,
                    "experiment_id": experiment_id,
                    "type": experiment["type"],
                    "metrics": experiment.get("results", {})
                })
                
                return {
                    "success": True,
                    "module": module_name,
                    "experiment_id": experiment_id,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to implement variant"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Implementation error: {str(e)}"
            }
            
    def get_experiment_status(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of experiments.
        
        Args:
            experiment_id: Optional specific experiment ID
            
        Returns:
            Experiment status information
        """
        if experiment_id:
            # Return specific experiment
            if experiment_id in self.current_experiments:
                return {
                    "success": True,
                    "experiment": self.current_experiments[experiment_id]
                }
            else:
                return {
                    "success": False,
                    "error": f"Experiment {experiment_id} not found"
                }
        else:
            # Return summary of all experiments
            experiment_summary = {}
            for exp_id, exp_data in self.current_experiments.items():
                experiment_summary[exp_id] = {
                    "module": exp_data["module"],
                    "type": exp_data["type"],
                    "status": exp_data["status"],
                    "timestamp": exp_data["timestamp"]
                }
                
            return {
                "success": True,
                "experiments": experiment_summary,
                "total": len(self.current_experiments)
            }
            
    def compare_variants(self, module_name: str, variant_paths: List[str]) -> Dict[str, Any]:
        """
        Compare multiple module variants.
        
        Args:
            module_name: Base module name
            variant_paths: List of variant paths to compare
            
        Returns:
            Comparison results
        """
        try:
            # Load original module
            original_code = self.code_repository.get_module(module_name)
            
            # Load variants
            variants = []
            for path in variant_paths:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        variants.append({
                            "path": path,
                            "code": f.read()
                        })
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error loading variant {path}: {str(e)}"
                    }
                    
            # Analyze each variant
            comparison = {
                "module": module_name,
                "original": {
                    "path": "original",
                    "size": len(original_code),
                    "lines": original_code.count('\n'),
                    "functions": len(re.findall(r'def\s+\w+\s*\(', original_code)),
                    "classes": len(re.findall(r'class\s+\w+', original_code))
                },
                "variants": []
            }
            
            for variant in variants:
                # Analyze variant
                var_data = {
                    "path": variant["path"],
                    "size": len(variant["code"]),
                    "lines": variant["code"].count('\n'),
                    "functions": len(re.findall(r'def\s+\w+\s*\(', variant["code"])),
                    "classes": len(re.findall(r'class\s+\w+', variant["code"])),
                    "diff_stats": {
                        "additions": 0,
                        "deletions": 0,
                        "changes": 0
                    }
                }
                
                # Generate diff
                original_lines = original_code.splitlines()
                variant_lines = variant["code"].splitlines()
                
                differ = difflib.Differ()
                diff = list(differ.compare(original_lines, variant_lines))
                
                # Count diff stats
                for line in diff:
                    if line.startswith('+ '):
                        var_data["diff_stats"]["additions"] += 1
                    elif line.startswith('- '):
                        var_data["diff_stats"]["deletions"] += 1
                    elif line.startswith('? '):
                        var_data["diff_stats"]["changes"] += 1
                        
                # Calculate similarity
                similarity = difflib.SequenceMatcher(None, original_code, variant["code"]).ratio()
                var_data["similarity"] = similarity
                
                comparison["variants"].append(var_data)
                
            return {
                "success": True,
                "comparison": comparison
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Comparison error: {str(e)}"
            }
            
    def cross_module_analysis(self, module_names: List[str]) -> Dict[str, Any]:
        """
        Analyze the relationships between multiple modules.
        
        Args:
            module_names: List of module names to analyze
            
        Returns:
            Cross-module analysis results
        """
        try:
            # Ensure architecture map is up to date
            if not self.architecture_map:
                self.build_full_architecture_map()
                
            # Extract relevant modules
            module_data = {}
            for module in module_names:
                if module in self.architecture_map:
                    module_data[module] = self.architecture_map[module]
                    
            if not module_data:
                return {
                    "success": False,
                    "error": "None of the specified modules found in architecture map"
                }
                
            # Analyze dependencies between modules
            dependencies = {}
            for module, data in module_data.items():
                dependencies[module] = []
                for dep in data.get("dependencies", []):
                    if dep in module_names:
                        dependencies[module].append(dep)
                        
            # Look for circular dependencies
            circular_deps = []
            for module in module_names:
                visited = set()
                path = [module]
                
                def dfs(current):
                    """Depth-first search to find cycles."""
                    if current in visited:
                        return False
                        
                    visited.add(current)
                    path.append(current)
                    
                    for dep in dependencies.get(current, []):
                        if dep in path:
                            # Found a cycle
                            cycle_start = path.index(dep)
                            circular_deps.append(path[cycle_start:] + [dep])
                            return True
                            
                        if dfs(dep):
                            return True
                            
                    path.pop()
                    return False
                    
                dfs(module)
                
            # Analyze shared classes and functions
            shared_definitions = {}
            for module, data in module_data.items():
                for class_name in data.get("classes", []):
                    if class_name not in shared_definitions:
                        shared_definitions[class_name] = []
                    shared_definitions[class_name].append(module)
                    
                for func_name in data.get("functions", []):
                    if func_name not in shared_definitions:
                        shared_definitions[func_name] = []
                    shared_definitions[func_name].append(module)
                    
            # Keep only shared definitions (in multiple modules)
            shared_definitions = {name: modules for name, modules in shared_definitions.items() if len(modules) > 1}
            
            # Generate recommendations
            recommendations = []
            
            # Recommend fixing circular dependencies
            if circular_deps:
                for cycle in circular_deps:
                    cycle_str = " -> ".join(cycle)
                    recommendations.append({
                        "type": "circular_dependency",
                        "modules": cycle,
                        "description": f"Circular dependency detected: {cycle_str}",
                        "suggestion": "Extract shared functionality into a separate module to break the circular dependency"
                    })
                    
            # Recommend consolidating shared definitions
            if shared_definitions:
                duplicated = []
                for name, modules in shared_definitions.items():
                    if len(modules) > 2:  # Only consider significant duplication
                        duplicated.append({
                            "name": name,
                            "modules": modules
                        })
                        
                if duplicated:
                    recommendations.append({
                        "type": "duplicated_definitions",
                        "duplicated": duplicated,
                        "description": f"Found {len(duplicated)} definitions duplicated across multiple modules",
                        "suggestion": "Consider refactoring to eliminate duplication and improve maintainability"
                    })
                    
            return {
                "success": True,
                "modules": module_names,
                "dependencies": dependencies,
                "circular_dependencies": circular_deps,
                "shared_definitions": shared_definitions,
                "recommendations": recommendations
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Cross-module analysis error: {str(e)}"
            }
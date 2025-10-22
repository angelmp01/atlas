"""
Test runner for ATLAS ML package.

This module provides a unified entry point for running all tests.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import test modules
from test_probability import TestProbabilityEstimation, TestProbabilityIntegration
from test_regressors import TestRegressionEstimators, TestRegressionIntegration
from test_featurization import TestFeatureEngineering, TestFeatureIntegration


def create_test_suite():
    """Create a comprehensive test suite for ATLAS ML package."""
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add probability tests
    suite.addTest(unittest.makeSuite(TestProbabilityEstimation))
    suite.addTest(unittest.makeSuite(TestProbabilityIntegration))
    
    # Add regression tests
    suite.addTest(unittest.makeSuite(TestRegressionEstimators))
    suite.addTest(unittest.makeSuite(TestRegressionIntegration))
    
    # Add feature engineering tests
    suite.addTest(unittest.makeSuite(TestFeatureEngineering))
    suite.addTest(unittest.makeSuite(TestFeatureIntegration))
    
    return suite


def run_tests(verbosity=2):
    """
    Run all tests with specified verbosity.
    
    Args:
        verbosity: Test output verbosity (0=minimal, 1=normal, 2=verbose)
        
    Returns:
        TestResult object
    """
    # Create test suite
    suite = create_test_suite()
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    
    # Run tests
    print("=" * 70)
    print("ATLAS ML Package Test Suite")
    print("=" * 70)
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception: ')[-1].split('\\n')[0]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    print("=" * 70)
    
    return result


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ATLAS ML package tests')
    parser.add_argument(
        '-v', '--verbosity', 
        type=int, 
        default=2, 
        choices=[0, 1, 2],
        help='Test output verbosity (0=minimal, 1=normal, 2=verbose)'
    )
    parser.add_argument(
        '--module', 
        type=str, 
        choices=['probability', 'regressors', 'featurization', 'all'],
        default='all',
        help='Specific test module to run'
    )
    
    args = parser.parse_args()
    
    # Run specific module or all tests
    if args.module == 'all':
        result = run_tests(args.verbosity)
    else:
        # Run specific module
        suite = unittest.TestSuite()
        
        if args.module == 'probability':
            suite.addTest(unittest.makeSuite(TestProbabilityEstimation))
            suite.addTest(unittest.makeSuite(TestProbabilityIntegration))
        elif args.module == 'regressors':
            suite.addTest(unittest.makeSuite(TestRegressionEstimators))
            suite.addTest(unittest.makeSuite(TestRegressionIntegration))
        elif args.module == 'featurization':
            suite.addTest(unittest.makeSuite(TestFeatureEngineering))
            suite.addTest(unittest.makeSuite(TestFeatureIntegration))
        
        runner = unittest.TextTestRunner(verbosity=args.verbosity)
        result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
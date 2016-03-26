using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace CocoWrapper
{
    using struct_pointer_t = System.Int64;
    using double_pointer_t = System.Int64;
    using void_pointer_t = System.Int64;
    using size_t = System.Int32;
    using cint = System.Int32;

    public class CocoLibraryWrapper
    {
        //Import funkcji z CocoLibrary.dll:


        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void hubert_coco_evaluate_function(struct_pointer_t problem, double[] x_dummy, double[] y, double[] x);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void hubert2_coco_evaluate_function(struct_pointer_t problem, double* x_dummy, double* y, double* x);



        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern char* coco_set_log_level(String level);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern struct_pointer_t coco_observer(String observer_name, String options);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern struct_pointer_t coco_problem_add_observer(struct_pointer_t problem, struct_pointer_t observer);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern struct_pointer_t coco_problem_remove_observer(struct_pointer_t problem, struct_pointer_t observer);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void coco_observer_free(struct_pointer_t observer);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern struct_pointer_t coco_suite(String suite_name, String suite_instance, String suite_options);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void coco_suite_free(struct_pointer_t suite);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern struct_pointer_t coco_suite_get_next_problem(struct_pointer_t suite, struct_pointer_t observer);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern struct_pointer_t coco_suite_get_problem(struct_pointer_t suite, size_t problem_index);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern size_t coco_problem_get_number_of_objectives(struct_pointer_t problem);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern double* coco_allocate_vector(size_t size);

        //[DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        //unsafe static extern void coco_evaluate_function(struct_pointer_t problem, double* x, double* y);
        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void coco_evaluate_function(struct_pointer_t problem, double[] x, double[] y);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void coco_free_memory(void* data);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern void coco_evaluate_constraint(struct_pointer_t problem, double[] x, double[] y);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern size_t coco_problem_get_dimension(struct_pointer_t problem);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern size_t coco_problem_get_number_of_constraints(struct_pointer_t problem);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern double* coco_problem_get_smallest_values_of_interest(struct_pointer_t problem);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern double* coco_problem_get_largest_values_of_interest(struct_pointer_t problem);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern char* coco_problem_get_id(struct_pointer_t problem);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern char* coco_problem_get_name(struct_pointer_t problem);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern size_t coco_problem_get_evaluations(struct_pointer_t problem);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern size_t coco_problem_get_suite_dep_index(struct_pointer_t problem);

        [DllImport("CocoLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        unsafe static extern cint coco_problem_final_target_hit(struct_pointer_t problem);



        //Właściwe opakowanie:

        /* Native methods */

        public static unsafe void cocoSetLogLevel(String logLevel)
        {
            coco_set_log_level(logLevel);
        }

        // Observer
        public static unsafe long cocoGetObserver(String observerName, String observerOptions)
        {
            return coco_observer(observerName, observerOptions);
        }

        public static unsafe void cocoFinalizeObserver(long observerPointer)
        {
            coco_observer_free(observerPointer);
        }

        public static unsafe long cocoProblemAddObserver(long problemPointer, long observerPointer)
        {
            return coco_problem_add_observer(problemPointer, observerPointer);
        }

        public static unsafe long cocoProblemRemoveObserver(long problemPointer, long observerPointer)
        {
            return coco_problem_remove_observer(problemPointer, observerPointer);
        }


        // Suite
        public static unsafe long cocoGetSuite(String suiteName, String suiteInstance, String suiteOptions)
        {
            return coco_suite(suiteName, suiteInstance, suiteOptions);
        }

        public static unsafe void cocoFinalizeSuite(long suitePointer)
        {
            coco_suite_free(suitePointer);
        }

        // Problem
        public static unsafe long cocoSuiteGetNextProblem(long suitePointer, long observerPointer)
        {
            return coco_suite_get_next_problem(suitePointer, observerPointer);
        }

        public unsafe long cocoSuiteGetProblem(long suitePointer, size_t problemIndex)
        {
            return coco_suite_get_problem(suitePointer, problemIndex);
        }

        // Functions     
        public static unsafe double[] cocoEvaluateFunction(long problemPointer, double[] x)
        {
            int number_of_objectives = (int)coco_problem_get_number_of_objectives(problemPointer);
            double* y = coco_allocate_vector(number_of_objectives);
            double[] ytab = new double[number_of_objectives];
            for (int i = 0; i < number_of_objectives; i++)
            {
                ytab[i] = y[i];
            }

            ////////
            int dim = cocoProblemGetDimension(problemPointer);
            double* xa = coco_allocate_vector(dim);
            double[] xtab = new double[dim];
            for (int i = 0; i < dim; i++)
            {
                xtab[i] = x[i];
                xa[i] = x[i];
            }
            ////////
            double ww = 88.9;
            double* z = coco_allocate_vector(number_of_objectives);
            //coco_evaluate_function(problemPointer, xtab, ytab);
            hubert2_coco_evaluate_function(problemPointer, y, xa, &ww);
            //hubert_coco_evaluate_function(problemPointer, y, ytab, xtab);

            coco_free_memory(y);

            return ytab;
        }

        public static unsafe double[] cocoEvaluateConstraint(long problemPointer, double[] x)
        {
            throw new NotImplementedException();
        }

        // Getters
        public static unsafe int cocoProblemGetDimension(long problemPointer)
        {
            return (int)coco_problem_get_dimension(problemPointer);
        }

        public static unsafe int cocoProblemGetNumberOfObjectives(long problemPointer)
        {
            return (int)coco_problem_get_number_of_objectives(problemPointer);
        }

        public static unsafe int cocoProblemGetNumberOfConstraints(long problemPointer)
        {
            return (int)coco_problem_get_number_of_constraints(problemPointer);
        }

        public static unsafe double[] cocoProblemGetSmallestValuesOfInterest(long problemPointer)
        {
            int dimension = (int)coco_problem_get_dimension(problemPointer);
            double* result = coco_problem_get_smallest_values_of_interest(problemPointer);
            double[] tab = new double[dimension];
            for (int i = 0; i < dimension; i++)
            {
                tab[i] = result[i];
            }

            return tab;
        }

        public static unsafe double[] cocoProblemGetLargestValuesOfInterest(long problemPointer)
        {
            int dimension = (int)coco_problem_get_dimension(problemPointer);
            double* result = coco_problem_get_largest_values_of_interest(problemPointer);
            double[] tab = new double[dimension];
            for (int i = 0; i < dimension; i++)
            {
                tab[i] = result[i];
            }

            return tab;
        }

        public static unsafe String cocoProblemGetId(long problemPointer)
        {
            char* str = coco_problem_get_id(problemPointer);

            return new string(str);
        }

        public static unsafe String cocoProblemGetName(long problemPointer)
        {
            char* str = coco_problem_get_name(problemPointer);

            return new string(str);
        }

        public static unsafe long cocoProblemGetEvaluations(long problemPointer)
        {
            return (long)coco_problem_get_evaluations(problemPointer);
        }

        public static unsafe long cocoProblemGetIndex(long problemPointer)
        {
            return (long)coco_problem_get_suite_dep_index(problemPointer);
        }

        public static unsafe int cocoProblemIsFinalTargetHit(long problemPointer)
        {
            return (int)coco_problem_final_target_hit(problemPointer);
        }
    }

    public class Observer
    {
        private long pointer; // Pointer to the coco_observer_t object
        private String name;


        public Observer(String observerName, String observerOptions)
        {
            try
            {
                this.pointer = CocoLibraryWrapper.cocoGetObserver(observerName, observerOptions);
                this.name = observerName;
            }
            catch (Exception e)
            {
                throw new Exception("Observer constructor failed.\n" + e.Message);
            }
        }

        public void finalizeObserver()
        {
            try
            {
                CocoLibraryWrapper.cocoFinalizeObserver(this.pointer);
            }
            catch (Exception e)
            {
                throw new Exception("Observer finalization failed.\n" + e.Message);
            }
        }

        public long getPointer()
        {
            return this.pointer;
        }

        public String getName()
        {
            return this.name;
        }

        public String toString()
        {
            return getName();
        }
    }

    public class Suite
    {

        private long pointer; // Pointer to the coco_suite_t object
        private String name;

        public Suite(String suiteName, String suiteInstance, String suiteOptions)
        {
            try
            {
                this.pointer = CocoLibraryWrapper.cocoGetSuite(suiteName, suiteInstance, suiteOptions);
                this.name = suiteName;
            }
            catch (Exception e)
            {
                throw new Exception("Suite constructor failed.\n" + e.Message);
            }
        }

        public void finalizeSuite()
        {
            try
            {
                CocoLibraryWrapper.cocoFinalizeSuite(this.pointer);
            }
            catch (Exception e)
            {
                throw new Exception("Suite finalization failed.\n" + e.Message);
            }
        }

        public long getPointer()
        {
            return this.pointer;
        }

        public String getName()
        {
            return this.name;
        }

        public String toString()
        {
            return getName();
        }
    }

    public class Benchmark
    {
        private Suite suite;
        private Observer observer;

        /** 
         * Constructor 
         */
        public Benchmark(Suite suite, Observer observer)
        {
            this.suite = suite;
            this.observer = observer;
        }

        /**
         * Function that returns the next problem in the suite. When it comes to the end of the suite, 
         * it returns null.
         * @return the next problem in the suite or null when there is no next problem  
         * @throws Exception 
         */
        public Problem getNextProblem()
        {

            try
            {
                long problemPointer = CocoLibraryWrapper.cocoSuiteGetNextProblem(suite.getPointer(), observer.getPointer());

                if (problemPointer == 0)
                    return null;

                return new Problem(problemPointer);
            }
            catch (Exception e)
            {
                throw new Exception("Fetching of next problem failed.\n" + e.Message);
            }
        }

        /**
         * Finalizes the observer and suite. This method needs to be explicitly called in order to log 
         * the last results.
         * @throws Exception 
         */
        public void finalizeBenchmark()
        {

            try
            {
                observer.finalizeObserver();
                suite.finalizeSuite();
            }
            catch (Exception e)
            {
                throw new Exception("Benchmark finalization failed.\n" + e.Message);
            }
        }
    }

    public class Problem
    {
        private long pointer; // Pointer to the coco_problem_t object

        private int dimension;
        private int number_of_objectives;
        private int number_of_constraints;

        private double[] lower_bounds;
        private double[] upper_bounds;

        private String id;
        private String name;

        private long index;

        public Problem(long pointer)
        {

            try
            {
                this.dimension = CocoLibraryWrapper.cocoProblemGetDimension(pointer);
                this.number_of_objectives = CocoLibraryWrapper.cocoProblemGetNumberOfObjectives(pointer);
                this.number_of_constraints = CocoLibraryWrapper.cocoProblemGetNumberOfConstraints(pointer);

                this.lower_bounds = CocoLibraryWrapper.cocoProblemGetSmallestValuesOfInterest(pointer);
                this.upper_bounds = CocoLibraryWrapper.cocoProblemGetLargestValuesOfInterest(pointer);

                this.id = CocoLibraryWrapper.cocoProblemGetId(pointer);
                this.name = CocoLibraryWrapper.cocoProblemGetName(pointer);

                this.index = CocoLibraryWrapper.cocoProblemGetIndex(pointer);

                this.pointer = pointer;
            }
            catch (Exception e)
            {
                throw new Exception("Problem constructor failed.\n" + e.Message);
            }
        }

        /**
         * Evaluates the function in point x and returns the result as an array of doubles. 
         * @param x
         * @return the result of the function evaluation in point x
         */
        public double[] evaluateFunction(double[] x)
        {
            return CocoLibraryWrapper.cocoEvaluateFunction(this.pointer, x);
        }

        /**
         * Evaluates the constraint in point x and returns the result as an array of doubles. 
         * @param x
         * @return the result of the constraint evaluation in point x
         */
        public double[] evaluateConstraint(double[] x)
        {
            return CocoLibraryWrapper.cocoEvaluateConstraint(this.pointer, x);
        }

        // Getters
        public long getPointer()
        {
            return this.pointer;
        }

        public int getDimension()
        {
            return this.dimension;
        }

        public int getNumberOfObjectives()
        {
            return this.number_of_objectives;
        }

        public int getNumberOfConstraints()
        {
            return this.number_of_constraints;
        }

        public double[] getSmallestValuesOfInterest()
        {
            return this.lower_bounds;
        }

        public double getSmallestValueOfInterest(int index)
        {
            return this.lower_bounds[index];
        }

        public double[] getLargestValuesOfInterest()
        {
            return this.upper_bounds;
        }

        public double getLargestValueOfInterest(int index)
        {
            return this.upper_bounds[index];
        }

        public String getId()
        {
            return this.id;
        }

        public String getName()
        {
            return this.name;
        }

        public long getEvaluations()
        {
            return CocoLibraryWrapper.cocoProblemGetEvaluations(pointer);
        }

        public long getIndex()
        {
            return this.index;
        }

        public Boolean isFinalTargetHit()
        {
            return (CocoLibraryWrapper.cocoProblemIsFinalTargetHit(pointer) == 1);
        }

        public String toString()
        {
            return this.getId();
        }
    }
}

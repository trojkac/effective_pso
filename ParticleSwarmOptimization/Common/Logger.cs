using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public interface ILogger
    {
        string CurrentLog { get; }
        void Log(ILogable obj);
        void Log(string logLine);
        void GenerateLog();
    }

    public interface ILogable
    {
        string ToLog();
    }

    public class ConsoleLogger : ILogger
    {
        public string CurrentLog
        {
            get { return logBuilder.ToString(); }
        }

        public StringBuilder logBuilder;

        public ConsoleLogger(string info)
        {
            logBuilder = new StringBuilder();
        }

        public void Log(ILogable obj)
        {
            var s = obj.ToLog();
            logBuilder.AppendLine(s);
            Console.WriteLine(s);
        }

        public void Log(string logLine)
        {
            Console.WriteLine(logLine);

            logBuilder.AppendLine(logLine);
        }

        public void GenerateLog()
        {
            Console.WriteLine(CurrentLog);
        }
    }
}

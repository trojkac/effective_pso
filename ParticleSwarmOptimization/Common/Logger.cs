using System;
using System.Collections.Generic;
using System.IO;
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
    public class FileLogger : ILogger, IDisposable
    {
        private StreamWriter _file;
        public string CurrentLog
        {
            get { return logBuilder.ToString(); }
        }

        public StringBuilder logBuilder;

        public FileLogger(string fileName)
        {
            logBuilder = new StringBuilder();
            _file =  new StreamWriter(new FileStream(fileName,FileMode.Create));
        }

        public void Log(ILogable obj)
        {
            var s = obj.ToLog();
            logBuilder.AppendLine(s);
            _file.WriteLine(s);
        }

        public void Log(string logLine)
        {
            logBuilder.AppendLine(logLine);
            _file.WriteLine(logLine);
        }

        public void GenerateLog()
        {
            _file.Close();
        }

        public void Dispose()
        {
            _file.Close();
            _file.Dispose();
        }
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

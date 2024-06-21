import java.io.PrintStream;
import java.lang.management.ManagementFactory;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;

import org.renaissance.Plugin;


public class JITServerPlugin implements Plugin, Plugin.AfterBenchmarkSetUpListener, Plugin.BeforeHarnessShutdownListener
{
	public JITServerPlugin(String[] args)
	{
		for (String arg : args) {
			if (arg.startsWith("-S=")) {
				sleepTimeMs = Integer.parseInt(arg.split("=")[1]);
			} else switch (arg) {
				case "-q": doSigQuit = true; break;
				case "-s": doSccStats = true; break;
			}
		}
	}

	@Override
	public void afterBenchmarkSetUp(String benchmark)
	{
		System.out.println(timeFormatter.format(Instant.now()) + " Setup complete");
	}

	@Override
	public void beforeHarnessShutdown()
	{
		if (sleepTimeMs > 0) {
			try {
				Thread.sleep(sleepTimeMs);
			} catch (Exception e) {
				System.err.println(e.toString());
			}
		}

		System.out.println(timeFormatter.format(Instant.now()) + " Benchmark complete");

		try {
			if (doSigQuit) {
				String pid = ManagementFactory.getRuntimeMXBean().getName().split("@")[0];
				new ProcessBuilder("kill", "-SIGQUIT", pid).inheritIO().start().waitFor();
			}

			if (doSccStats) {
				String sccArgs = "-Xshareclasses:printStats,name=renaissance,cacheDir=/output/.classCache";
				new ProcessBuilder("/opt/ibm/java/bin/java", sccArgs).inheritIO().start().waitFor();
			}

		} catch (Exception e) {
			System.err.println(e.toString());
		}
	}

	private int sleepTimeMs = 0;
	private boolean doSigQuit = false;
	private boolean doSccStats = false;

	private DateTimeFormatter timeFormatter = new DateTimeFormatterBuilder().appendInstant(6).toFormatter();
}

import pstats
import io

def analyze_profile(profile_file):
    stream = io.StringIO()
    p = pstats.Stats(profile_file, stream=stream)
    
    p.sort_stats('cumulative').print_stats(10)

    print(stream.getvalue())


if __name__ == "__main__":
    analyze_profile('app_profile.prof')
    analyze_profile('rs_profile.prof')
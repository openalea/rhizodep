from data_utility.data_analysis import analyze_data

def test_just_analyze():
    analyze_data(outputs_dirpath="test/outputs", 
                 on_sums=True,
                 on_performance=True,
                 target_properties=[]
                 )
    
test_just_analyze()
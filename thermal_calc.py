from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.psychrometrics import v_relative

label_names = ["Cold", "Cool", "Slightly Cool", "Neutral", "Slightly Warm", "Warm", "Hot"]

# measured air velocity
v = 0.1

v_r = v_relative(v=v, met=1.0)
print(v_r)

for temp in range(10, 37):
    for humid in range(0, 101):
        pmv_index = 0
        
        # print("Temp: {0:.2f} | Humidity: {1:.2f}".format(temp, humid))

        # calculate PMV in accordance with the ASHRAE 55 2017
        results = pmv_ppd(ta=temp, tr=temp, vr=v_r, rh=humid, met=1.0, clo=0.6, wme=0, standard="ashrae")

        # print the results
        # print(results)
        if(results["pmv"] < -2.5):
            pmv_index = 0
        elif(results["pmv"] >= -2.5 and results["pmv"] < -1.5):
            pmv_index = 1
        elif(results["pmv"] >= -1.5 and results["pmv"] < -0.5):
            pmv_index = 2
        elif(results["pmv"] >= -0.5 and results["pmv"] < 0.5):
            pmv_index = 3
        elif(results["pmv"] >= 0.5 and results["pmv"] < 1.5):
            pmv_index = 4
        elif(results["pmv"] >= 1.5 and results["pmv"] < 2.5):
            pmv_index = 5
        elif(results["pmv"] >= 2.5):
            pmv_index = 6

        # print PMV value
        # print("ASHRAE PMV: {0:.2f} | {1}".format(results['pmv'], label_names[pmv_index]))
        print("1013.25, 0.1, {0:.2f}, 1.0, 0.6, {1:.2f}, {2}".format(humid, temp, pmv_index))
        # print(results)

        # for users who wants to use the IP system
        # results_ip = pmv_ppd(ta=temp, tr=temp, vr=0.4, rh=humid, met=1.0, clo=0.6, units="IP")
        # print(results_ip)
        # print("ISO PMV: {0:.2f} ({1}) | {2}".format(results['pmv'], label_names[int(results['pmv']) + 3]), results)
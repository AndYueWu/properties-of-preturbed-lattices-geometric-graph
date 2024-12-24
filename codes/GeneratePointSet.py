import PLGG

Length = 100
sd = 0.2

pointset1 = PLGG.PointSet(Length,sd)
pointset1.save_distance_matrix("L=100_1")
pointset2 = PLGG.PointSet(Length,sd)
pointset1.save_distance_matrix("L=100_2")
pointset3 = PLGG.PointSet(Length,sd)
pointset1.save_distance_matrix("L=100_3")


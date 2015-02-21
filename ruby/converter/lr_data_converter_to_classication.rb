#!/usr/bin/ruby

fin = "/home/kzk/datasets/spark-sample/lr_data.txt"
fout = "/home/kzk/datasets/spark-sample/svm_data.txt"
open(fout, "w") do |fpout|
  open(fin, "r").each do |l|
    sl = l.chomp.split(" ")
    if sl[0] == "-1" then
      sl[0] = "0"
    end
    fpout.puts(sl.join(" "))
  end
end

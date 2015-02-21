require 'rubygems'
require 'rubystats'

norm = Rubystats::NormalDistribution.new(10, 2)
cdf = norm.cdf(11)
pdf = norm.pdf(11)
puts "CDF(11): #{cdf}"
puts "PDF(11): #{pdf}"

puts 

puts "Random numbers from normal distribution:"
10.times do
  puts norm.rng
end

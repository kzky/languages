require "rubygems"
require "active_record"
require 'logger'

config = YAML.load_file("./blog_db.yml")
ActiveRecord::Base.establish_connection(config["db"]["development"])
#ActiveRecord::Base.logger = Logger.new(STDOUT); ## for debugging

class Post < ActiveRecord::Base
  #scope :top3, order("created_at").limit(3) # deprecated
  scope :top3, -> {order("created_at").limit(3)}
end


# delete: record-oriented fast
# destory: object-oriented slow

Post.where(:id => 1..2).delete_all()
p Post.all

Post.find(3).destroy()
p Post.all




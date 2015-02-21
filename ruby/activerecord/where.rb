require "rubygems"
require "active_record"
require 'logger'

config = YAML.load_file("./blog_db.yml")
ActiveRecord::Base.establish_connection(config["db"]["development"])
ActiveRecord::Base.logger = Logger.new(STDOUT); ## for debugging

class Post < ActiveRecord::Base
  #scope :top3, order("created_at").limit(3) # deprecated
  scope :top3, -> {order("created_at").limit(3)}
end

# where clause
p Post.where(:title => "title_1", :id => 1)
p Post.where("title = ? and id = ?", "title_1", 1)
p Post.where("title = :title and id = :id", {:title => "title_1", :id =>1})
p Post.where("id > ?", 2)
p Post.where("body like ?", "body%")
p Post.where(:id => 1..3)
p Post.where(:id => [1, 4])
p Post.order("id desc").limit(3)

# scope
p Post.top3

# first_or_create
Post.first_or_create(:title => "title_4");
p Post.order("id desc").limit(3)

Post.where(:title => "title_5").first_or_create;
p Post.order("id desc").limit(3)

Post.where(:title => "title_6").first_or_create do |p|
  p.body = "body_6";
end
p Post.order("id desc").limit(3)



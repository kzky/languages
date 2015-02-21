#!/usr/bin/ruby
# -*- coding: utf-8 -*-
require 'socket'

class EtherHeader
    attr_reader :ether_dhost
    attr_reader :ether_shost
    attr_reader :ether_type

    def initialize(frame)
        @ether_dhost = mac_tos(frame, 0)
        @ether_shost = mac_tos(frame, 6)
        @ether_type = (frame[12] << 8) + frame[13]
    end

    def mac_tos(frame, index)
        return sprintf('%02X:%02X:%02X:%02X:%02X:%02X',
            frame[index], frame[index + 1], frame[index + 2], frame[index + 3], frame[index + 4], frame[index + 5])
    end
end

class IPHeader
    attr_reader :version
    attr_reader :ip_hl
    attr_reader :ip_tos
    attr_reader :ip_len
    attr_reader :ip_id
    attr_reader :ip_off 
    attr_reader :ip_ttl
    attr_reader :ip_p
    attr_reader :ip_sum
    attr_reader :ip_src
    attr_reader :ip_dst

    def initialize(packet, index)
        @version = (packet[index] >> 4) & 0xF
        @ip_hl = packet[index] & 0xF
        @ip_tos = packet[index + 1]
        @ip_len = (packet[index + 2] << 8) + packet[index + 3]
        @ip_id = (packet[index + 4] << 8) + packet[index + 5]
        @ip_off = (packet[index + 6] << 8) + packet[index + 7]
        @ip_ttl = packet[index + 8]
        @ip_p = packet[index + 9]
        @ip_sum = (packet[index + 10] << 8) + packet[index + 11]
        @ip_src = ip_tos(packet, index + 12)
        @ip_dst = ip_tos(packet, index + 16)
    end

    def ip_tos(packet, index)
        return sprintf("%d.%d.%d.%d", packet[index], packet[index + 1], packet[index + 2], packet[index + 3])
    end
end

ETH_P_ALL    = 0x0300
ETHERTYPE_IP = 0x800

socket = Socket.open(Socket::PF_INET, Socket::SOCK_PACKET, ETH_P_ALL)

buff = socket.read(8192)
ether_header = EtherHeader.new(buff)


puts 'Ethernetフレーム'
puts "送信元MACアドレス #{ether_header.ether_shost}"
puts "送信先MACアドレス #{ether_header.ether_dhost}"
puts sprintf("EtherType = 0x%X", ether_header.ether_type)

if(ether_header.ether_type == ETHERTYPE_IP)
    puts 'IPパケット'
    ip_header = IPHeader.new(buff, 14)
    puts "送信元IPアドレス #{ip_header.ip_src}"
    puts "送信先IPアドレス #{ip_header.ip_dst}"
end

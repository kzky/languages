# -*- coding: utf-8 -*-
from django.conf.urls.defaults import *
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.template import RequestContext, Context, loader
from django.core.urlresolvers import reverse
from django.shortcuts import render_to_response, get_object_or_404
from mysite.polls.models import Poll
from django.http import Http404

def index(request):
    latest_poll_list = Poll.objects.all().order_by('-pub_date')[:5]
#    t = loader.get_template('mysite/polls/index.html')
#    c = Context({  ## passed to template as key
#        'latest_poll_list': latest_poll_list,
#    })
#    return HttpResponse(t.render(c))
    return render_to_response('mysite/polls/index.html',
                              {'latest_poll_list': latest_poll_list})

def detail(request, poll_id):
    #    try:
    #        p = Poll.objects.get(pk=poll_id)
    #    except Poll.DoesNotExist:
    #        raise Http404
    #    return render_to_response('mysite/polls/detail.html', {'poll': p})
    p = get_object_or_404(Poll, pk=poll_id) ## shortcut function
    return render_to_response('mysite/polls/detail.html', {'poll': p})

def vote(request, poll_id):
    p = get_object_or_404(Poll, pk=poll_id)
    try:
        selected_choice = p.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Poll 投票フォームを再表示します。
        return render_to_response('mysite/polls/detail.html',
                                  RequestContext (request,
                                                  {'poll': p,
                                                   'error_message': "選択肢を選んでいません。",
                                                   }))
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # ユーザが Back ボタンを押して同じフォームを提出するのを防ぐ
        # ため、POST データを処理できた場合には、必ず
        # HttpResponseRedirect を返すようにします。
        return HttpResponseRedirect(reverse('mysite.polls.views.results', args=(p.id,)))

def results(request, poll_id):
    p = get_object_or_404(Poll, pk=poll_id)
    return render_to_response('mysite/polls/results.html', {'poll': p})

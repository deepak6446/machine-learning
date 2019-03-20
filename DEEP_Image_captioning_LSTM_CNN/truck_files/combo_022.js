YUI.add("notification-event-photo-comment-mention-models",function(e){function t(e){t.superclass.constructor.call(this,e)}e.Models[this.name]=t,e.extend(t,e.FlickrModelRegistry,{name:this.name,attributes:{type:{},owner:{isModel:!0},timestamp:{}}})},"@VERSION@",{requires:["flickr-model-registry"]});YUI.add("notification-event-photo-group-invite-models",function(e){function t(e){t.superclass.constructor.call(this,e)}e.Models[this.name]=t,e.extend(t,e.FlickrModelRegistry,{name:this.name,attributes:{type:{},owner:{isModel:!0},group:{id:"",name:""},timestamp:{}}})},"@VERSION@",{requires:["flickr-model-registry"]});YUI.add("notification-event-group-models",function(e){function t(e){t.superclass.constructor.call(this,e)}e.Models[this.name]=t,e.extend(t,e.FlickrModelRegistry,{name:this.name,attributes:{type:{},owner:{isModel:!0},timestamp:{}}})},"@VERSION@",{requires:["flickr-model-registry"]});YUI.add("notification-event-note-models",function(e){function t(e){t.superclass.constructor.call(this,e)}e.Models[this.name]=t,e.extend(t,e.FlickrModelRegistry,{name:this.name,attributes:{type:{},owner:{isModel:!0},timestamp:{}}})},"@VERSION@",{requires:["flickr-model-registry"]});YUI.add("notification-event-share-models",function(e){function t(e){t.superclass.constructor.call(this,e)}e.Models[this.name]=t,e.extend(t,e.FlickrModelRegistry,{name:this.name,attributes:{type:{},owner:{isModel:!0},timestamp:{}}})},"@VERSION@",{requires:["flickr-model-registry"]});YUI.add("notification-event-group-reply-models",function(e){function t(e){t.superclass.constructor.call(this,e)}e.Models[this.name]=t,e.extend(t,e.FlickrModelRegistry,{name:this.name,attributes:{type:{},owner:{isModel:!0},timestamp:{},replyId:{}}})},"@VERSION@",{requires:["flickr-model-registry"]});YUI.add("notification-event-flickrmail-models",function(e){function i(e){i.superclass.constructor.call(this,e)}e.Models[this.name]=i,e.extend(i,e.FlickrModelRegistry,{name:this.name,attributes:{type:{},owner:{isModel:!0},mail:{},timestamp:{}}})},"@VERSION@",{requires:["flickr-model-registry"]});YUI.add("notification-event-testimonial-models",function(e){function t(e){t.superclass.constructor.call(this,e)}e.Models[this.name]=t,e.extend(t,e.FlickrModelRegistry,{name:this.name,attributes:{type:{},owner:{isModel:!0},timestamp:{},url:{}}})},"@VERSION@",{requires:["flickr-model-registry"]});YUI.add("flickr-notifications-markread-updater",function(r,i){"use strict";r.namespace("ModelUpdaters")["flickr-notifications-markread"]={run:function(e,a){return a.callAPI("flickr.notifications.markread",e).then(null,r.FetcherErrorLogger(i))}}},"@VERSION@",{requires:["flickr-promise"]});YUI.add("flickr-activity-getNotificationsCount-fetcher",function(t,i){"use strict";t.namespace("ModelFetchers")["flickr-activity-getNotificationsCount"]={run:function(e,o){var n=this;return t.FlickrPromise({apiResponse:o.callAPI("flickr.activity.getNotificationsCount",e),personNotifsModelRegistry:o.getModelRegistry("person-notifications-models")}).then(function(t){return n._processResponse(t,o,e)},t.FetcherErrorLogger(i))},_processResponse:function(t,i,e){var o=t.apiResponse;return t.personNotifsModelRegistry.setValue(e.id,"unseenNotificationCount",o.activity.count),o.activity.count}}},"@VERSION@",{requires:["flickr-promise"],optional:["person-notifications-models","api-helper"]});YUI.add("flickr-activity-muteObject-updater",function(e,t){"use strict";e.namespace("ModelUpdaters")["flickr-activity-muteObject"]={run:function(r,c){var i=r;return c.callAPI("flickr.activity.muteObjectByType",i).then(null,e.FetcherErrorLogger(t))}}},"@VERSION@",{requires:["flickr-promise"]});YUI.add("flickr-activity-unmuteObject-updater",function(e,t){"use strict";e.namespace("ModelUpdaters")["flickr-activity-unmuteObject"]={run:function(r,c){var i=r;return c.callAPI("flickr.activity.unmuteObjectByType",i).then(null,e.FetcherErrorLogger(t))}}},"@VERSION@",{requires:["flickr-promise"]});YUI.add("person-notifications-models",function(e){function t(e){t.superclass.constructor.call(this,e)}e.Models[this.name]=t,e.extend(t,e.FlickrModelRegistry,{name:this.name,remote:{read:function(t){return e.ModelFetchers["flickr-people-getInfo"].run(t,this.appContext)},readUnseenNotificationCount:function(t,r){return e.ModelFetchers["flickr-activity-getNotificationsCount"].run(t,r)},updateMutedObjects:function(t,r){return e.ModelUpdaters["flickr-activity-muteObject"].run(t,r)},updateUnmutedObjects:function(t,r){return e.ModelUpdaters["flickr-activity-unmuteObject"].run(t,r)}},remoteReadUnseenNotificationCount:function(e,t,r){return this.setValue(e,"unseenLastFetchedTimestamp",Date.now()),this.remote.readUnseenNotificationCount(t,r)},remoteUpdateMutedObjects:function(e,t,r){var n=this;return this.remote.updateMutedObjects(t,r).then(function(t){return n.setValue(e,"muteChanged",!0),t})},remoteUpdateUnmutedObjects:function(e,t,r){var n=this;return this.remote.updateUnmutedObjects(t,r).then(function(t){return n.setValue(e,"muteChanged",!0),t})},attributes:{id:{setter:function(e){return e||this.appContext.getViewer().nsid}},unseenNotificationCount:{validator:function(t){return e.AttributeHelpers.validateInteger(t)},setter:function(t){return e.AttributeHelpers.coerceInteger(t)},defaultValue:0},unreadFlickrMailCount:{validator:function(t){return e.AttributeHelpers.validateInteger(t)},setter:function(t){return e.AttributeHelpers.coerceInteger(t)},defaultValue:0},unseenLastFetchedTimestamp:{validator:function(t){return e.AttributeHelpers.validateInteger(t)},setter:function(t){return e.AttributeHelpers.coerceInteger(t)},defaultValue:0},muteChanged:{validator:function(t){return e.AttributeHelpers.validateBoolean(t)},setter:function(t){return e.AttributeHelpers.coerceBoolean(t)},defaultValue:!1},notifications:{isCollection:!0,pageFetch:{listFetcher:e.ListFetchers["flickr-activity-recentByType"]}}}})},"@VERSION@",{requires:["flickr-model-registry","flickr-activity-recentByType-fetcher","flickr-notifications-markread-updater","flickr-people-getInfo-fetcher","flickr-activity-getNotificationsCount-fetcher","flickr-activity-muteObject-updater","flickr-activity-unmuteObject-updater","attribute-helpers"]});YUI.add("flickr-activity-recentByType-fetcher",function(e,t){"use strict";e.namespace("ListFetchers")["flickr-activity-recentByType"]={run:function(o,i){var r=this,s=this._processParams(o);return new e.FlickrPromise({apiResponse:i.callAPI("flickr.activity.recentByType",s),notificationModelRegistry:i.getModelRegistry("notification-models"),notificationsModelRegistry:i.getModelRegistry("person-notifications-models"),groupInfoModelRegistry:i.getModelRegistry("group-info-models"),groupModelRegistry:i.getModelRegistry("group-models"),personModelRegistry:i.getModelRegistry("person-models"),photoModelRegistry:i.getModelRegistry("photo-models"),contactModelRegistry:i.getModelRegistry("contact-models"),photoEngagementModelRegistry:i.getModelRegistry("photo-engagement-models"),personRelationshipModelRegistry:i.getModelRegistry("person-relationship-models"),photoStatsRegistry:i.getModelRegistry("photo-stats-models"),discussionModelRegistry:i.getModelRegistry("notification-group-discussion-models"),galleryModelRegistry:i.getModelRegistry("notification-gallery-models"),flickrMailModelRegistry:i.getModelRegistry("notification-flickrmail-models"),faveEventModelRegistry:i.getModelRegistry("notification-event-fave-models"),commentEventModelRegistry:i.getModelRegistry("notification-event-comment-models"),followerEventModelRegistry:i.getModelRegistry("notification-event-follower-models"),addedToGalleryEventModelRegistry:i.getModelRegistry("notification-event-added-to-gallery-models"),tagEventModelRegistry:i.getModelRegistry("notification-event-tag-models"),peopleEventModelRegistry:i.getModelRegistry("notification-event-people-models"),photoCommentMentionNotifRegisty:i.getModelRegistry("notification-event-photo-comment-mention-models"),photoGroupInviteNotifRegistry:i.getModelRegistry("notification-event-photo-group-invite-models"),groupNotifModelRegistry:i.getModelRegistry("notification-event-group-models"),noteEventModelRegistry:i.getModelRegistry("notification-event-note-models"),shareEventModelRegistry:i.getModelRegistry("notification-event-share-models"),groupReplyEventModelRegistry:i.getModelRegistry("notification-event-group-reply-models"),friendJoinEventModelRegistry:i.getModelRegistry("notification-event-friend-join-models"),groupTopicNewEventModelRegistry:i.getModelRegistry("notification-event-group-topic-new-models"),flickrMailEventModelRegistry:i.getModelRegistry("notification-event-flickrmail-models"),testimonialEventModelRegistry:i.getModelRegistry("notification-event-testimonial-models")}).then(function(e){return r._processResponse(e,o,s,i)},e.FetcherErrorLogger(t))},_processParams:function(t){var o,i={extras:["sizes","icon_urls","ignored","rev_ignored","tags","autotags","datecreate"].join(",")+","+e.APIHelper.request.getRebootPhotoExtras(),per_page:t.perPage||30,count:30,page:t.page||1,per_object:50};return o=e.merge(i,t.apiParams||{}),YUI.Env.isServer&&e.SearchHelper.unescapeParameters(o),o},_processResponse:function(t,o,i,r){var s,n,l,a=t.apiResponse,d=t.notificationModelRegistry,p=t.notificationsModelRegistry,g=t.photoModelRegistry,c=t.personModelRegistry,y=t.contactModelRegistry,m=t.groupModelRegistry,f=(t.personRelationshipModelRegistry,t.photoEngagementModelRegistry),R=t.photoStatsRegistry,u=t.discussionModelRegistry,M=t.galleryModelRegistry,v=t.flickrMailModelRegistry,h=[];return s={fave:t.faveEventModelRegistry,comment:t.commentEventModelRegistry,contacted:t.followerEventModelRegistry,gallery:t.addedToGalleryEventModelRegistry,tag:t.tagEventModelRegistry,people:t.peopleEventModelRegistry,photo_comment_mention:t.photoCommentMentionNotifRegisty,group_invite:t.photoGroupInviteNotifRegistry,group_user_invite:t.groupNotifModelRegistry,note:t.noteEventModelRegistry,share:t.shareEventModelRegistry,reply_reply:t.groupReplyEventModelRegistry,topic_reply:t.groupReplyEventModelRegistry,friend_join:t.friendJoinEventModelRegistry,group_topic_new:t.groupTopicNewEventModelRegistry,flickrmail:t.flickrMailEventModelRegistry,testimonial:t.testimonialEventModelRegistry},a.items.item.forEach(function(t){var o,i,n,a,p=t.activity?t.activity.event:[];if(o={objectType:t.type,seen:1===t.seen,id:e.guid(),events:[],isMuted:1===t.is_muted},"photo"===t.type){var E=e.clone(t);if(E.title){var _=E.title.content||E.title._content;E.title=_}o.object=e.APIHelper.response.parsePhotos({photos:[E],photoModelRegistry:g,personModelRegistry:c,photoEngagementModelRegistry:f,photoStatsModelRegistry:R}).shift()}else"person"===t.type?o.object=c.exists(t.id)?c.proxy(t.id):c.add(e.APIHelper.response.parsePerson({nsid:t.user,pathAlias:t.pathAlias,username:t.username,realname:t.realname,isAdFree:t.isAdFree,ispro:t.ispro,proBadge:t.proBadge,datecreate:t.datecreate,iconurls:t.iconurls})):"group"===t.type?o.object=m.exists(t.id)?m.proxy(t.id):m.add(m.parseGroupModel(t.id,t.pathAlias,t)):"discussion"===t.type?((n=e.clone(t)).id=t.group_id,n.title=t.group_name,a=m.exists(t.group_id)?m.proxy(t.group_id):m.add(m.parseGroupModel(t.group_id,t.pathAlias,n)),o.object=u.exists(t.id)?u.proxy(t.id):u.add(u.parseDiscussionModel(t.id,t.pathAlias,t,a))):"gallery"===t.type?o.object=M.addOrUpdate(M.parseGalleryModel(t)):"flickrmail"===t.type&&(o.object=v.addOrUpdate(v.parseFlickrMailModel(t)));p.forEach(function(n){var a,d,p;if(n.type in s){if("people"===n.type)l={},d=e.APIHelper.response.parseContact({nsid:n.person,username:n.personname}),p=e.APIHelper.response.parseContact({nsid:n.user,username:n.realname}),l.personTagged=y.exists(d.id)?y.proxy(d.id):y.add(d),l.tagger=y.exists(p.id)?y.proxy(p.id):y.add(p);else if("group_invite"===n.type)n.name=n.groupName,n.id=n.groupId,n.iconurls=n.group_iconurls,l=m.exists(n.id)?m.proxy(n.id):m.add(m.parseGroupModel(n.groupId,n.pathAlias,n));else{if("note"===n.type&&!r.flipper.isFlipped("enable-scrappy-notes"))return;if("gallery"===n.type)l={id:n.galleryid,title:n.title,url:"/photos/"+n.owner+"/galleries/"+n.galleryid,iconurls:n.gallery_iconurls};else if("flickrmail"===n.type)l={mail:{id:t.id,title:t.title,url:"/mail/"+t.id},owner:{nsid:t.user,username:t.username,realname:t.realname,iconserver:t.iconserver,iconfarm:t.iconfarm,pathAlias:t.pathalias,iconUrls:t.iconurls}};else if("testimonial"===n.type){var g=c.proxy(r.getViewer().nsid),f=e.URLHelper.generatePersonUrls(g),R=(f.profile,f.profile.slice(0,f.profile.length-1)+"#testimonial"+t.id);l={id:t.id,owner:{nsid:t.user,username:t.username,realname:t.realname,iconserver:t.iconserver,iconfarm:t.iconfarm,pathAlias:t.pathalias,iconUrls:t.iconurls},url:R}}}if(a=e.APIHelper.response.parseNotificationsEvents[n.type](n,l),a.owner=y.addOrUpdate(a.owner),a.owner.getValue("isMe")&&"group_user_invite"!==n.type)return;i=s[n.type].add(a),o.events.push(i)}}),d.exists(o.id)||h.push(d.add(o))}),n={notifications:h},p.exists(o.id)?p.getValue(o.id,"notifications").addPage({page:i.page,perPage:i.per_page,pageContent:h,totalItems:parseInt(a.items.total||0,10)}):p.add(n),h}}},"@VERSION@",{requires:["flickr-promise","search-helper","querystring-parse-simple"],optional:["url-helper","notification-models","person-notifications-models","contact-models","photo-models","notification-group-discussion-models","notification-gallery-models","notification-flickrmail-models","notification-event-fave-models","notification-event-comment-models","notification-event-follower-models","notification-event-added-to-gallery-models","notification-event-people-models","notification-event-tag-models","notification-event-photo-comment-mention-models","notification-event-photo-group-invite-models","notification-event-group-models","notification-event-note-models","notification-event-share-models","notification-event-group-reply-models","notification-event-flickrmail-models","notification-event-testimonial-models"]});YUI.add("notification-models",function(e){function t(e){t.superclass.constructor.call(this,e)}e.Models[this.name]=t,e.extend(t,e.FlickrModelRegistry,{name:this.name,remote:{read:function(t){return e.ListFetchers["flickr-activity-recentByType"].run(t,this.appContext)}},attributes:{id:{setter:function(e){return e||this.appContext.getViewer().nsid}},seen:{},objectType:{},object:{isModel:!0},events:{isListProxy:!0},isMuted:{}}})},"@VERSION@",{requires:["flickr-model-registry","flickr-activity-recentByType-fetcher"]});YUI.add("hermes-template-notification-photo-extra-info",function(e,a){var n=e.Template.Handlebars.revive({1:function(e,a,n,t,i){return e.escapeExpression((n.intlMessage||a&&a.intlMessage||n.helperMissing).call(null!=a?a:{},{name:"intlMessage",hash:{time:null!=a?a.time:a,intlName:null!=a?a.intlName:a},data:i}))},compiler:[7,">= 4.0.0"],main:function(e,a,n,t,i){var l;return'<span class="extra-info">\n\t'+(null!=(l=(n.friendlySinceDate||a&&a.friendlySinceDate||n.helperMissing).call(null!=a?a:{},null!=(l=null!=a?a.event:a)?l.timestamp:l,!0,!0,{name:"friendlySinceDate",hash:{},fn:e.program(1,i,0),inverse:e.noop,data:i}))?l:"")+"\n</span>"},useData:!0}),t={};e.Array.each([],function(a){var n=e.Template.get("hermes/"+a);n&&(t[a]=n)}),e.Template.register("hermes/notification-photo-extra-info",function(a,i){return i=i||{},i.partials=i.partials?e.merge(t,i.partials):t,n(a,i)})},"@VERSION@",{requires:["template-base","handlebars-base"]});YUI.add("hermes-template-notification-photo-comment",function(l,n){var e=l.Template.Handlebars.revive({1:function(l,n,e,a,t){var o;return null!=(o=e.if.call(null!=n?n:{},null!=n?n.subPeopleCount:n,{name:"if",hash:{},fn:l.program(2,t,0),inverse:l.program(4,t,0),data:t}))?o:""},2:function(l,n,e,a,t){var o;return"\t\t\t\t"+l.escapeExpression((e.intlHTMLMessage||n&&n.intlHTMLMessage||e.helperMissing).call(null!=n?n:{},{name:"intlHTMLMessage",hash:{mediaType:null!=(o=null!=n?n.object:n)?o.mediaType:o,objectUrl:null!=(o=null!=n?n.object:n)?o.url:o,secondPersonUrl:null!=(o=null!=(o=null!=n?n.people:n)?o[1]:o)?o.url:o,secondPersonName:null!=(o=null!=(o=null!=n?n.people:n)?o[1]:o)?o.name:o,firstPersonUrl:null!=(o=null!=(o=null!=n?n.people:n)?o[0]:o)?o.url:o,firstPersonName:null!=(o=null!=(o=null!=n?n.people:n)?o[0]:o)?o.name:o,count:null!=n?n.subPeopleCount:n,intlName:"notifications.NOTIF_PHOTO_COMMENT_YOURS_MULTI"},data:t}))+"\n"},4:function(l,n,e,a,t){var o;return"\t\t\t\t"+l.escapeExpression((e.intlHTMLMessage||n&&n.intlHTMLMessage||e.helperMissing).call(null!=n?n:{},{name:"intlHTMLMessage",hash:{mediaType:null!=(o=null!=n?n.object:n)?o.mediaType:o,objectUrl:null!=(o=null!=n?n.object:n)?o.url:o,personUrl:null!=(o=null!=(o=null!=n?n.people:n)?o[0]:o)?o.url:o,personName:null!=(o=null!=(o=null!=n?n.people:n)?o[0]:o)?o.name:o,intlName:"notifications.NOTIF_PHOTO_COMMENT_YOURS"},data:t}))+"\n"},6:function(l,n,e,a,t){var o;return null!=(o=e.if.call(null!=n?n:{},null!=n?n.subPeopleCount:n,{name:"if",hash:{},fn:l.program(7,t,0),inverse:l.program(9,t,0),data:t}))?o:""},7:function(l,n,e,a,t){var o;return"\t\t\t\t"+l.escapeExpression((e.intlHTMLMessage||n&&n.intlHTMLMessage||e.helperMissing).call(null!=n?n:{},{name:"intlHTMLMessage",hash:{ownerName:null!=(o=null!=(o=null!=n?n.object:n)?o.owner:o)?o.displayname:o,ownerUrl:null!=(o=null!=(o=null!=n?n.object:n)?o.owner:o)?o.url:o,mediaType:null!=(o=null!=n?n.object:n)?o.mediaType:o,objectUrl:null!=(o=null!=n?n.object:n)?o.url:o,secondPersonUrl:null!=(o=null!=(o=null!=n?n.people:n)?o[1]:o)?o.url:o,secondPersonName:null!=(o=null!=(o=null!=n?n.people:n)?o[1]:o)?o.name:o,firstPersonUrl:null!=(o=null!=(o=null!=n?n.people:n)?o[0]:o)?o.url:o,firstPersonName:null!=(o=null!=(o=null!=n?n.people:n)?o[0]:o)?o.name:o,count:null!=n?n.subPeopleCount:n,intlName:"notifications.NOTIF_PHOTO_COMMENT_OTHER_MULTI"},data:t}))+"\n"},9:function(l,n,e,a,t){var o;return"\t\t\t\t"+l.escapeExpression((e.intlHTMLMessage||n&&n.intlHTMLMessage||e.helperMissing).call(null!=n?n:{},{name:"intlHTMLMessage",hash:{ownerName:null!=(o=null!=(o=null!=n?n.object:n)?o.owner:o)?o.displayname:o,ownerUrl:null!=(o=null!=(o=null!=n?n.object:n)?o.owner:o)?o.url:o,mediaType:null!=(o=null!=n?n.object:n)?o.mediaType:o,objectUrl:null!=(o=null!=n?n.object:n)?o.url:o,personUrl:null!=(o=null!=(o=null!=n?n.people:n)?o[0]:o)?o.url:o,personName:null!=(o=null!=(o=null!=n?n.people:n)?o[0]:o)?o.name:o,intlName:"notifications.NOTIF_PHOTO_COMMENT_OTHER"},data:t}))+"\n"},compiler:[7,">= 4.0.0"],main:function(l,n,e,a,t){var o;return'<div class="details">\n\t<span class="headline">\n'+(null!=(o=e.if.call(null!=n?n:{},null!=(o=null!=(o=null!=n?n.object:n)?o.owner:o)?o.isMe:o,{name:"if",hash:{},fn:l.program(1,t,0),inverse:l.program(6,t,0),data:t}))?o:"")+"\t</span>\n\n"+(null!=(o=l.invokePartial(a["notification-photo-extra-info"],n,{name:"notification-photo-extra-info",data:t,indent:"\t",helpers:e,partials:a,decorators:l.decorators}))?o:"")+"</div>"},usePartial:!0,useData:!0}),a={};l.Array.each(["notification-photo-extra-info"],function(n){var e=l.Template.get("hermes/"+n);e&&(a[n]=e)}),l.Template.register("hermes/notification-photo-comment",function(n,t){return t=t||{},t.partials=t.partials?l.merge(a,t.partials):a,e(n,t)})},"@VERSION@",{requires:["template-base","handlebars-base","hermes-template-notification-photo-extra-info"]});